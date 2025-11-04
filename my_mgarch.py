# my_mgarch.py
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

class mgarch:
    """
    25일마다만 재적합(파라미터 갱신), 그 사이엔 update_one으로
    관측 수익률과 하루 앞 평균예측을 넣어 상태(h, Q) 갱신 → t+1 공분산 반환.
    """

    def __init__(self, dist='norm'):
        if dist not in ('norm', 't'):
            raise ValueError("dist must be 'norm' or 't'.")
        self.dist = dist

    # ---------- 단변량 GARCH(1,1) ----------
    def garch_fit(self, returns_1d):
        r = np.asarray(returns_1d, dtype=float).ravel()
        res = minimize(self.garch_loglike, x0=(1e-2, 1e-2, 0.94),
                       args=(r,), bounds=((1e-12, 10.0), (1e-12, 1.0), (1e-12, 1.0)))
        return res.x

    def garch_loglike(self, params, r):
        omega, alpha, beta = params
        T = r.shape[0]
        var = np.empty(T, dtype=float)
        var[0] = max(r[0]**2, 1e-8)
        for t in range(1, T):
            var[t] = omega + alpha * r[t-1]**2 + beta * var[t-1]
        var = np.clip(var, 1e-12, None)
        ll = -0.5 * (np.log(2*np.pi*var) + (r**2)/var)
        return -np.sum(ll)

    def garch_var(self, params, returns_1d):
        r = np.asarray(returns_1d, dtype=float).ravel()
        omega, alpha, beta = params
        T = r.shape[0]
        var = np.empty(T, dtype=float)
        var[0] = max(r[0]**2, 1e-8)
        for t in range(1, T):
            var[t] = omega + alpha * r[t-1]**2 + beta * var[t-1]
        return var

    # ---------- 수치 안정 보조 ----------
    @staticmethod
    def _to_spd(M, eps=1e-10):
        M = 0.5 * (M + M.T)
        w = np.linalg.eigvalsh(M)
        m = w.min()
        if m < eps:
            M = M + np.eye(M.shape[0]) * (eps - m + 1e-12)
        return M

    # ---------- DCC 우도(정규/학생-t) ----------
    def mgarch_loglike(self, params, D_path):
        a, b = params
        if not (0 <= a <= 1 and 0 <= b <= 1 and a + b < 1):
            return np.inf
        X = self.rt
        Z = X / (D_path + 1e-300)
        S = np.corrcoef(Z.T)
        S = np.nan_to_num(0.5*(S + S.T))
        # 초기 Q
        Q = S.copy()
        N = self.N
        ll = 0.0
        for t in range(1, self.T):
            z = Z[t-1]
            Q = (1 - a - b) * S + a * np.outer(z, z) + b * Q
            d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
            R = Q / (d[:, None] * d[None, :] + 1e-300)
            R = 0.5 * (R + R.T)
            signR, logdetR = np.linalg.slogdet(R)
            if signR <= 0:
                return np.inf
            u = Z[t]                       # r_t / D_t
            q = u @ np.linalg.solve(R, u)
            ll += -0.5*(N*np.log(2*np.pi) + logdetR + q)
        return -ll

    def mgarch_logliket(self, params, D_path):
        a, b, dof = params
        if dof <= 2: return np.inf
        if not (0 <= a <= 1 and 0 <= b <= 1 and a + b < 1):
            return np.inf
        X = self.rt
        Z = X / (D_path + 1e-300)
        S = np.corrcoef(Z.T)
        S = np.nan_to_num(0.5*(S + S.T))
        Q = S.copy()
        N = self.N
        const = gammaln((N + dof)/2) - gammaln(dof/2) - (N/2)*np.log((dof - 2)*np.pi)
        ll = 0.0
        for t in range(1, self.T):
            zprev = Z[t-1]
            Q = (1 - a - b) * S + a * np.outer(zprev, zprev) + b * Q
            d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
            R = Q / (d[:, None] * d[None, :] + 1e-300)
            R = 0.5 * (R + R.T)
            signR, logdetR = np.linalg.slogdet(R)
            if signR <= 0:
                return np.inf
            u = Z[t]
            q = u @ np.linalg.solve(R, u)
            ll += const - 0.5*logdetR - ((dof + N)/2)*np.log(1 + q/(dof - 2))
        return -ll

    # ---------- 적합: 파라미터 추정 + 상태 초기화 ----------
    def fit(self, returns):
        X = np.asarray(returns, dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("returns must be 2D with N>=2 assets")
        self.rt = X - X.mean(axis=0, keepdims=True)  # 평균 제거(평균예측은 update_one 인자로)
        self.T, self.N = self.rt.shape

        # 개별 GARCH 파라미터 및 표준편차 경로
        D_path = np.zeros_like(self.rt)
        self.omega = np.zeros(self.N); self.alpha = np.zeros(self.N); self.beta = np.zeros(self.N)
        var_last = np.zeros(self.N)
        for i in range(self.N):
            p = self.garch_fit(self.rt[:, i])
            v = self.garch_var(p, self.rt[:, i])
            D_path[:, i] = np.sqrt(np.clip(v, 1e-12, None))
            self.omega[i], self.alpha[i], self.beta[i] = p
            var_last[i] = v[-1]
        self.h_last = np.clip(var_last, 1e-12, None)
        self.D_last = np.sqrt(self.h_last)

        # DCC 파라미터 추정
        if self.dist == 'norm':
            res = minimize(self.mgarch_loglike, x0=(0.02, 0.97), args=(D_path,),
                           bounds=((1e-6, 1-1e-6), (1e-6, 1-1e-6)))
            self.a, self.b = res.x
        else:
            res = minimize(self.mgarch_logliket, x0=(0.02, 0.97, 6.0), args=(D_path,),
                           bounds=((1e-6, 1-1e-6), (1e-6, 1-1e-6), (2.1, None)))
            self.a, self.b, self.dof = res.x

        # 장기상관 S와 초기 Q/H
        Z = self.rt / (D_path + 1e-300)
        S = np.corrcoef(Z.T); S = np.nan_to_num(0.5*(S + S.T))
        self.S = self._to_spd(S, eps=1e-8)
        self.Q_last = self.S.copy()

        qd = np.sqrt(np.clip(np.diag(self.Q_last), 1e-12, None))
        R_last = self.Q_last / (qd[:, None] * qd[None, :] + 1e-300)
        R_last = 0.5 * (R_last + R_last.T)
        H_last = (self.D_last[:, None] * R_last * self.D_last[None, :])
        self.H_last = self._to_spd(H_last, eps=1e-10)

        return {'mu': X.mean(axis=0), 'a': self.a, 'b': self.b} if self.dist == 'norm' \
               else {'mu': X.mean(axis=0), 'a': self.a, 'b': self.b, 'dof': self.dof}

    # ---------- 하루 업데이트(핵심): 관측 + 평균예측 → 혁신 → 상태 갱신 ----------
    def update_one(self, r_t, mean_pred=None):
        """
        r_t: shape (N,) 관측 수익률 (원시값; 평균 제거하지 말 것)
        mean_pred: shape (N,) m_{t|t-1} (없으면 0으로 처리)
        반환: {'dist': ..., 'cov': H_{t+1|t}}
        """
        if not hasattr(self, 'h_last'):
            raise RuntimeError("Call fit(...) before update_one(...)")
        r_t = np.asarray(r_t, dtype=float).ravel()
        if r_t.shape[0] != self.N:
            raise ValueError("r_t shape mismatch")

        m = 0.0 if mean_pred is None else np.asarray(mean_pred, dtype=float).ravel()
        if np.isscalar(m): m = np.full(self.N, m)

        eps_t = r_t - m                                # 혁신
        D_prev = np.sqrt(np.clip(self.h_last, 1e-12, None))
        z = eps_t / (D_prev + 1e-300)                  # 표준화 혁신

        # DCC 재귀
        self.Q_last = (1 - self.a - self.b) * self.S + self.a * np.outer(z, z) + self.b * self.Q_last
        qd = np.sqrt(np.clip(np.diag(self.Q_last), 1e-12, None))
        R_next = self.Q_last / (qd[:, None] * qd[None, :] + 1e-300)
        R_next = 0.5 * (R_next + R_next.T)

        # 개별 분산 재귀(GARCH 1,1)
        self.h_last = self.omega + self.alpha * (eps_t**2) + self.beta * self.h_last
        self.h_last = np.clip(self.h_last, 1e-12, None)
        D_next = np.sqrt(self.h_last)

        H_next = (D_next[:, None] * R_next * D_next[None, :])
        H_next = self._to_spd(H_next, eps=1e-10)
        self.D_last = D_next
        self.H_last = H_next
        return {'dist': self.dist, 'cov': H_next}

    # ---------- 현재 상태 기반 1-스텝 예측 ----------
    def predict(self, ndays=1):
        if not hasattr(self, 'H_last'):
            raise RuntimeError("Call fit(...) first.")
        if ndays != 1:
            return {'dist': self.dist, 'cov': self.H_last * np.sqrt(ndays)}
        return {'dist': self.dist, 'cov': self.H_last}
