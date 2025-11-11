# ===============================================================
# ppo_meters_fast.py — PPO 版 (Dirichlet Policy, 無 WLS, 單一 UT)
# ===============================================================

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

# ---------------- Environment ----------------
class BeamEnv:
    def __init__(self, N_sats=10, beam_radius=50, UT_position=(0.0, 0.0)):
        self.N_sats = N_sats
        self.beam_radius = beam_radius
        self.UT_position = np.array(UT_position, dtype=np.float32)
        self.reset()

    def reset(self):
        self.detected_beams = self._generate_beam_positions()
        self.undetected_beams = self._generate_beam_positions()
        self.state = self._compute_state()
        return self.state

    def _generate_beam_positions(self):
        center_offset = np.random.uniform(-10, 10, (self.N_sats, 2)).astype(np.float32)
        start_points = self.UT_position + center_offset + np.random.uniform(-5, 5, (self.N_sats, 2)).astype(np.float32)
        end_points = start_points + np.random.uniform(-3, 3, (self.N_sats, 2)).astype(np.float32)
        return np.stack([start_points, end_points], axis=1)

    def _beam_center(self, beam):
        return (beam[0] + beam[1]) / 2.0

    def _compute_state(self):
        feats = []
        for i in range(self.N_sats):
            c = self._beam_center(self.detected_beams[i])
            d = float(np.linalg.norm(c - self.UT_position))
            feats.append(np.clip(1.0 - d / 20.0, 0.0, 1.0))
        return np.array(feats, dtype=np.float32)

    def _circle_poly(self, center, radius, n_points=32):
        theta = np.linspace(0, 2*np.pi, n_points)
        pts = np.stack([center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta)], axis=1)
        return Polygon(pts)

    def _intersection_centroid(self):
        det_polys = [self._circle_poly(self._beam_center(b), self.beam_radius/5.0) for b in self.detected_beams]
        inter = cascaded_union(det_polys)
        und_polys = [self._circle_poly(self._beam_center(b), self.beam_radius/5.0) for b in self.undetected_beams]
        for p in und_polys:
            if inter.is_empty: break
            if inter.intersects(p):
                inter = inter.difference(p)
        if not inter.is_empty:
            c = inter.centroid
            return np.array([c.x, c.y], dtype=np.float32)
        return self.UT_position.copy()

    def _weighted_cog(self, weights):
        centers = np.array([self._beam_center(b) for b in self.detected_beams], dtype=np.float32)
        w = np.clip(weights, 1e-8, 1.0)
        w /= np.sum(w)
        return np.sum(centers * w[:, None], axis=0)

    def _error_km(self, pos_est):
        return float(np.linalg.norm(pos_est - self.UT_position))

    def step(self, action):
        pos_est = self._weighted_cog(action)
        error_km = self._error_km(pos_est)
        reward = -np.exp(error_km / 10.0) + 1.0
        self.detected_beams = self._generate_beam_positions()
        self.undetected_beams = self._generate_beam_positions()
        self.state = self._compute_state()
        done = (error_km < 0.001) or (np.random.rand() > 0.98)
        return self.state, float(reward), bool(done), {"error_km": error_km}

    def visualize(self, weights=None, episode=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        det_polys = [self._circle_poly(self._beam_center(b), self.beam_radius/5.0) for b in self.detected_beams]
        inter = cascaded_union(det_polys)
        und_polys = [self._circle_poly(self._beam_center(b), self.beam_radius/5.0) for b in self.undetected_beams]
        for p in und_polys:
            if inter.is_empty: break
            if inter.intersects(p): inter = inter.difference(p)
        if not inter.is_empty:
            if inter.geom_type == 'Polygon':
                x,y = inter.exterior.xy; ax1.fill(x,y, alpha=0.3, fc='gray', ec='none')
            else:
                for g in inter.geoms:
                    x,y = g.exterior.xy; ax1.fill(x,y, alpha=0.3, fc='gray', ec='none')
        base_pos = self._intersection_centroid()
        base_err_m = self._error_km(base_pos)*1000.0
        ax1.plot(base_pos[0], base_pos[1], '*', color='yellow', markersize=18)
        ax1.text(0.02, 0.98, f'Error: {base_err_m:.2f} m', transform=ax1.transAxes, va='top')

        if weights is not None:
            pos_ai = self._weighted_cog(weights)
            err_ai_m = self._error_km(pos_ai)*1000.0
            ax2.plot(pos_ai[0], pos_ai[1], '*', color='yellow', markersize=18)
            ax2.text(0.02, 0.98, f'Error: {err_ai_m:.2f} m', transform=ax2.transAxes, va='top')

        for i in range(self.N_sats):
            bc_det = self._beam_center(self.detected_beams[i])
            bc_und = self._beam_center(self.undetected_beams[i])
            w = weights[i] if weights is not None else 1.0
            lw_det = 1.5 + 2.5 * w
            lw_und = 1.5 + 2.5 * (1.0 - w)
            for ax in (ax1, ax2):
                ax.add_patch(Circle(bc_det, self.beam_radius/5.0, color='blue', alpha=0.6, fill=False, linewidth=lw_det))
                ax.add_patch(Circle(bc_und, self.beam_radius/5.0, color='red', alpha=0.6, fill=False, linewidth=lw_und))
        for ax in (ax1, ax2):
            ax.plot(self.UT_position[0], self.UT_position[1], 'ko', markerfacecolor='none', markersize=10)
            ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
            ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if episode is not None: plt.suptitle(f'Episode {episode}', y=1.02)
        plt.close(fig)


# ---------------- PPO Core ----------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.conc_head = nn.Linear(hidden, action_dim)
        self.softplus = nn.Softplus()
    def forward(self, x):
        h = self.net(x)
        return self.softplus(self.conc_head(h)) + 1.0
    def dist(self, x):
        return Dirichlet(self.forward(x))

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, ent=0.01, vcoef=0.5, device='cpu'):
        self.pi = PolicyNet(state_dim, action_dim).to(device)
        self.v = ValueNet(state_dim).to(device)
        self.opt = optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=lr)
        self.gamma, self.lam, self.clip, self.ent, self.vcoef = gamma, lam, clip, ent, vcoef
        self.device = device

    @torch.no_grad()
    def act(self, s_np):
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
        d = self.pi.dist(s)
        a = d.rsample()
        logp = d.log_prob(a)
        val = self.v(s)
        return a.cpu().numpy(), float(logp.cpu().numpy()), float(val.cpu().numpy())

    def evaluate(self, s, a):
        d = self.pi.dist(s)
        logp = d.log_prob(a); ent = d.entropy(); val = self.v(s)
        return logp, ent, val

    def update(self, S, A, old_logp, R, Adv, epochs=10, mbsize=128):
        S = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(A, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        R = torch.as_tensor(R, dtype=torch.float32, device=self.device)
        Adv = torch.as_tensor(Adv, dtype=torch.float32, device=self.device)
        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)
        N = S.size(0); idx = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for st in range(0, N, mbsize):
                mb = idx[st:st+mbsize]
                logp, ent, val = self.evaluate(S[mb], A[mb])
                ratio = torch.exp(logp - old_logp[mb])
                s1, s2 = ratio*Adv[mb], torch.clamp(ratio, 1-self.clip, 1+self.clip)*Adv[mb]
                loss_pi = -torch.min(s1, s2).mean()
                loss_v = nn.MSELoss()(val, R[mb])
                loss = loss_pi + self.vcoef*loss_v - self.ent*ent.mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()


# ---------------- Training ----------------
def train_ppo(total_episodes=1000, rollout_steps=1024
              , update_epochs=10, minibatch_size=128, device='cpu'):
    start_time = time.time()
    env = BeamEnv()
    state_dim, action_dim = 10, 10
    agent = PPO(state_dim, action_dim, device=device)

    ep_rewards, ep_errors_m, weights_hist = [], [], []
    last_vis_weights = np.ones(action_dim)/action_dim

    for ep in range(total_episodes):
        s = env.reset(); done=False
        S,A,LP,R,D,V = [],[],[],[],[],[]
        total_r=0; errs=[]
        for _ in range(rollout_steps):
            a, lp, v = agent.act(s)
            ns, r, done, info = env.step(a)
            S.append(s); A.append(a); LP.append(lp); R.append(r); D.append(float(done)); V.append(v)
            s = ns; total_r += r; errs.append(info['error_km']*1000.0)
            last_vis_weights = a
            if done: s = env.reset(); done=False
        last_v = agent.v(torch.as_tensor(s, dtype=torch.float32, device=device)).item()
        V_t = np.array(V); nextV = np.concatenate([V_t[1:], np.array([last_v])]); R_t = np.array(R); D_t = np.array(D)
        adv, gae = [],0.0
        for t in reversed(range(len(R_t))):
            delta = R_t[t] + agent.gamma*(1-D_t[t])*nextV[t] - V_t[t]
            gae = delta + agent.gamma*agent.lam*(1-D_t[t])*gae
            adv.insert(0, gae)
        adv=np.array(adv); ret=V_t+adv
        agent.update(np.array(S), np.array(A), np.array(LP), ret, adv, epochs=update_epochs, mbsize=minibatch_size)
        ep_rewards.append(total_r); ep_errors_m.append(np.mean(errs)); weights_hist.append(last_vis_weights/np.sum(last_vis_weights))

    # 最後一次視覺化
    env.visualize(weights=last_vis_weights/np.sum(last_vis_weights), episode=total_episodes-1)

    end=time.time(); elapsed=end-start_time
    pf=2*10*128+2*128*128+2*128*10; vf=2*10*128+2*128*128+2*128*1
    mf=pf+vf; rf=total_episodes*rollout_steps*mf
    mb=int(np.ceil(rollout_steps/minibatch_size))*update_epochs*total_episodes; uf=3*mb*mf; tf=rf+uf

    print('========== PPO 訓練摘要 ==========')
    print(f'[Model FLOPs] 每次 forward 約 {mf/1e6:.2f} M FLOPs')
    print(f'[Training FLOPs] Rollout 累計 ≈ {rf/1e9:.2f} GFLOPs')
    print(f'[Training FLOPs] Update 累計 ≈ {uf/1e9:.2f} GFLOPs')
    print(f'[TOTAL FLOPs] 模型×訓練總次數 ≈ {tf/1e9:.2f} GFLOPs')
    print(f'[Training Time] 總耗時: {elapsed:.2f} 秒')
    if ep_errors_m: print(f'[Final Error] 定位誤差距離: {ep_errors_m[-1]:.2f} m')

    plt.figure(figsize=(12,5))

    # --- Reward Curve ---
    plt.subplot(1,2,1)
    plt.plot(ep_rewards, label='Raw Reward', color='steelblue')
    if len(ep_rewards) > 5:
        w = max(5, len(ep_rewards)//20)
        ma = np.convolve(ep_rewards, np.ones(w)/w, mode='valid')
        plt.plot(np.arange(w-1, len(ma)+w-1), ma, color='orange', linewidth=2, label=f'Smoothed Reward')
    plt.xlabel('Episodes',fontsize=18)
    plt.ylabel('Reward',fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- Error Curve ---
    plt.subplot(1,2,2)
    plt.plot(ep_errors_m, label='Raw Error', color='steelblue')
    if len(ep_errors_m) > 5:
        w2 = max(5, len(ep_errors_m)//20)
        ma2 = np.convolve(ep_errors_m, np.ones(w2)/w2, mode='valid')
        plt.plot(np.arange(w2-1, len(ma2)+w2-1), ma2, color='orange', linewidth=2, label=f'Smoothed Error')
    plt.xlabel('Episodes',fontsize=18)
    plt.ylabel('Error (m)',fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    w=weights_hist[-1] if weights_hist else last_vis_weights/np.sum(last_vis_weights)
    plt.bar(range(len(w)),w); plt.xlabel('Beam'); plt.ylabel('Weight'); plt.title('Final Beam Weight Distribution'); plt.grid(); plt.show()

    if len(weights_hist)>0:
        W=np.array(weights_hist)
        plt.figure(figsize=(12,4.5))
        im=plt.imshow(W.T,aspect='auto',origin='lower'); plt.colorbar(im,fraction=0.046,pad=0.04,label='Weight')
        plt.yticks(range(W.shape[1]),[f'Beam {i}' for i in range(W.shape[1])])
        plt.xlabel('Episode'); plt.title('Beam Weights Heatmap'); plt.tight_layout(); plt.show()

    return ep_rewards, ep_errors_m, env

if __name__=='__main__':
    print('Starting PPO training (single UT, fast mode)...')
    train_ppo(device='cpu')
