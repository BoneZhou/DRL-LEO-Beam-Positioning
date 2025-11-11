# ddqn_meters_fast.py — Double DQN final (single UT, no WLS)
# See header inside for details.
import time, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

class BeamEnv:
    def __init__(self, N_sats=10, beam_radius=50, UT_position=(0.0,0.0)):
        self.N_sats=N_sats; self.beam_radius=beam_radius
        self.UT_position=np.array(UT_position,dtype=np.float32); self.reset()
    def reset(self):
        self.detected_beams=self._gen(); self.undetected_beams=self._gen(); self.state=self._state(); return self.state
    def _gen(self):
        off=np.random.uniform(-10,10,(self.N_sats,2)).astype(np.float32)
        s=self.UT_position+off+np.random.uniform(-5,5,(self.N_sats,2)).astype(np.float32)
        e=s+np.random.uniform(-3,3,(self.N_sats,2)).astype(np.float32)
        return np.stack([s,e],axis=1)
    def _center(self,b): return (b[0]+b[1])/2.0
    def _state(self):
        feats=[]
        for i in range(self.N_sats):
            c=self._center(self.detected_beams[i]); d=float(np.linalg.norm(c-self.UT_position)); feats.append(np.clip(1.0-d/20.0,0.0,1.0))
        return np.array(feats,dtype=np.float32)
    def _circle(self,c,r,n=32):
        th=np.linspace(0,2*np.pi,n); pts=np.stack([c[0]+r*np.cos(th), c[1]+r*np.sin(th)],axis=1); return Polygon(pts)
    def _intersection_centroid(self):
        det=[self._circle(self._center(b), self.beam_radius/5.0) for b in self.detected_beams]
        inter=cascaded_union(det); und=[self._circle(self._center(b), self.beam_radius/5.0) for b in self.undetected_beams]
        for p in und:
            if inter.is_empty: break
            if inter.intersects(p): inter=inter.difference(p)
        if not inter.is_empty:
            c=inter.centroid; return np.array([c.x,c.y],dtype=np.float32)
        return self.UT_position.copy()
    def _weighted_cog(self,w):
        centers=np.array([self._center(b) for b in self.detected_beams],dtype=np.float32); w=np.clip(w,1e-8,1.0); w=w/(np.sum(w)+1e-8); return np.sum(centers*w[:,None],axis=0)
    def _err_km(self,p): return float(np.linalg.norm(p-self.UT_position))

class QNet(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__(); self.net=nn.Sequential(nn.Linear(sd,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,ad))
    def forward(self,x): return self.net(x)

class Replay:
    def __init__(self,cap=200_000): self.cap=cap; self.mem=[]; self.i=0
    def push(self,s,a,r,s2,d):
        if len(self.mem)<self.cap: self.mem.append(None)
        self.mem[self.i]=(s.astype(np.float32),int(a),float(r),s2.astype(np.float32),float(d)); self.i=(self.i+1)%self.cap
    def sample(self,b):
        batch=random.sample(self.mem,b); s,a,r,s2,d=zip(*batch); return np.array(s),np.array(a),np.array(r),np.array(s2),np.array(d)
    def __len__(self): return len(self.mem)

def softmax_w(q,t=1.0):
    z=(q-np.max(q))/max(t,1e-6); e=np.exp(z); return e/(np.sum(e)+1e-8)

def train_ddqn(total_episodes=1000, rollout_steps=512, batch_size=128, gamma=0.99, lr=3e-4, eps_start=1.0, eps_end=0.05, eps_decay=0.995, target_sync=500, device='cpu'):
    st=time.time(); env=BeamEnv(); sd=ad=env.N_sats
    q=QNet(sd,ad).to(device); tgt=QNet(sd,ad).to(device); tgt.load_state_dict(q.state_dict()); opt=optim.Adam(q.parameters(),lr=lr); buf=Replay(200_000)
    eps=eps_start; steps=0; epR=[]; epE=[]; Wh=[]
    last_w=np.ones(ad)/ad
    for ep in range(total_episodes):
        s=env.reset(); R=0.0; errs=[]
        for t in range(rollout_steps):
            with torch.no_grad(): qs=q(torch.as_tensor(s,dtype=torch.float32,device=device)).cpu().numpy()
            a = random.randrange(ad) if random.random()<eps else int(np.argmax(qs))
            w = softmax_w(qs); last_w=w.copy()
            pos=env._weighted_cog(w); err_km=env._err_km(pos); r=-np.exp(err_km/10.0)+1.0
            s2=env.reset(); d=1.0 if (err_km<0.001 or random.random()>0.98) else 0.0
            buf.push(s,a,r,s2,d); s=s2; R+=r; errs.append(err_km*1000.0)
            if len(buf)>=batch_size:
                sb,ab,rb,s2b,db = buf.sample(batch_size)
                sb=torch.as_tensor(sb,dtype=torch.float32,device=device); ab=torch.as_tensor(ab,dtype=torch.long,device=device).unsqueeze(-1)
                rb=torch.as_tensor(rb,dtype=torch.float32,device=device).unsqueeze(-1); s2b=torch.as_tensor(s2b,dtype=torch.float32,device=device); db=torch.as_tensor(db,dtype=torch.float32,device=device).unsqueeze(-1)
                with torch.no_grad():
                    qn=q(s2b); a2=torch.argmax(qn,dim=1,keepdim=True); yt=tgt(s2b).gather(1,a2); y=rb+(1.0-db)*gamma*yt
                qsa=q(sb).gather(1,ab); loss=nn.MSELoss()(qsa,y); opt.zero_grad(); loss.backward(); opt.step()
                steps += 1
                if steps % target_sync == 0:
                    tgt.load_state_dict(q.state_dict())
            if d>0.5: break
        eps=max(eps_end, eps*eps_decay); epR.append(R); epE.append(float(np.mean(errs) if len(errs)>0 else np.nan)); Wh.append(last_w/(np.sum(last_w)+1e-8))

    # Final localization figure
    final_w = Wh[-1] if len(Wh)>0 else (np.ones(ad)/ad)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,6))
    det=[env._circle(env._center(b), env.beam_radius/5.0) for b in env.detected_beams]
    inter=cascaded_union(det)
    und=[env._circle(env._center(b), env.beam_radius/5.0) for b in env.undetected_beams]
    for p in und:
        if inter.is_empty: break
        if inter.intersects(p): inter=inter.difference(p)
    if not inter.is_empty:
        if inter.geom_type=='Polygon':
            x,y=inter.exterior.xy; ax1.fill(x,y,alpha=0.3,fc='gray',ec='none')
        else:
            for g in inter.geoms:
                x,y=g.exterior.xy; ax1.fill(x,y,alpha=0.3,fc='gray',ec='none')
    base=env._intersection_centroid(); base_err=env._err_km(base)*1000.0
    ax1.plot(base[0],base[1],'*',color='yellow',markersize=18); ax1.text(0.02,0.98,f'Distance Error: {base_err:.2f} m',transform=ax1.transAxes,va='top'); ax1.set_title('ALG.B',bbox=dict(facecolor='white',edgecolor='black',pad=5))
    pos_ai=env._weighted_cog(final_w); err_ai=env._err_km(pos_ai)*1000.0
    ax2.plot(pos_ai[0],pos_ai[1],'*',color='yellow',markersize=18); ax2.text(0.02,0.98,f'Distance Error: {err_ai:.2f} m',transform=ax2.transAxes,va='top'); ax2.set_title('DDQN',bbox=dict(facecolor='white',edgecolor='black',pad=5))
    for i in range(env.N_sats):
        bc_det=env._center(env.detected_beams[i]); bc_und=env._center(env.undetected_beams[i]); w=final_w[i]; lw1=1.5+2.5*w; lw2=1.5+2.5*(1.0-w)
        for ax in (ax1,ax2):
            ax.add_patch(Circle(bc_det, env.beam_radius/5.0, color='blue', alpha=0.6, fill=False, linewidth=lw1))
            ax.add_patch(Circle(bc_und, env.beam_radius/5.0, color='red',  alpha=0.6, fill=False, linewidth=lw2))
    for ax in (ax1,ax2):
        ax.plot(env.UT_position[0], env.UT_position[1], 'ko', markerfacecolor='none', markersize=10)
        ax.set_xlim(-20,20); ax.set_ylim(-20,20); ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.grid(True,linestyle='--',alpha=0.7)
    plt.suptitle(f'Training Episode {total_episodes-1}'); plt.tight_layout(); plt.show()

    # Stats
    et=time.time()-st
    q_flops=2*sd*128+2*128*128+2*128*ad
    rollout=total_episodes*rollout_steps*q_flops
    updates=total_episodes*rollout_steps
    update=3*updates*q_flops
    total=rollout+update
    print('========== DDQN 訓練摘要 ==========\n'
          f'[Model FLOPs] QNet 每次 forward 約 {q_flops/1e6:.2f} M FLOPs\n'
          f'[Training FLOPs] Rollout 累計 ≈ {rollout/1e9:.2f} GFLOPs\n'
          f'[Training FLOPs] Update 累計 ≈ {update/1e9:.2f} GFLOPs\n'
          f'[TOTAL FLOPs] 模型×訓練總次數 ≈ {total/1e9:.2f} GFLOPs\n'
          f'[Training Time] 總耗時: {et:.2f} 秒\n'
          f'[Final Error] 定位誤差距離: {epE[-1]:.2f} m' if len(epE)>0 and not np.isnan(epE[-1]) else '')

       # Curves
    plt.figure(figsize=(12,5))

    # --- Reward Curve ---
    plt.subplot(1,2,1)
    plt.plot(epR, label='Raw Reward', color='steelblue')
    if len(epR) > 5:
        w = max(5, len(epR)//20)
        ma = np.convolve(epR, np.ones(w)/w, mode='valid')
        plt.plot(np.arange(w-1, w-1+len(ma)), ma, color='orange', linewidth=2,
                 label=f'Smoothed Reward')
    plt.xlabel('Episodes',fontsize=18)
    plt.ylabel('Reward',fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- Error Curve ---
    plt.subplot(1,2,2)
    plt.plot(epE, label='Raw Error', color='steelblue')
    if len(epE) > 5:
        w2 = max(5, len(epE)//20)
        ma2 = np.convolve(epE, np.ones(w2)/w2, mode='valid')
        plt.plot(np.arange(w2-1, w2-1+len(ma2)), ma2, color='orange', linewidth=2,
                 label=f'Smoothed Error')
    plt.xlabel('Episodes',fontsize=18)
    plt.ylabel('Error (m)',fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4)); fw=final_w; plt.bar(range(len(fw)),fw); plt.xlabel('Beam'); plt.ylabel('Weight'); plt.title('Final Beam Weight Distribution (DDQN)'); plt.grid(); plt.show()

    if len(Wh)>0:
        W=np.array(Wh); plt.figure(figsize=(12,4.5)); im=plt.imshow(W.T,aspect='auto',origin='lower'); plt.colorbar(im,fraction=0.046,pad=0.04,label='Weight')
        plt.yticks(range(W.shape[1]),[f'Beam {i}' for i in range(W.shape[1])]); plt.xlabel('Episode'); plt.title('Beam Weights over Training (Heatmap)'); plt.tight_layout(); plt.show()

    return epR, epE

if __name__=='__main__':
    train_ddqn(total_episodes=1000, rollout_steps=512, device='cpu')
