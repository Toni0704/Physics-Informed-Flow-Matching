# Residuals used for constraints projection in PCFM 

import torch

class Residuals:
    def __init__(self, data, x, t_grid, dx=None, dt=None, nx=None, nt=None, rho=None, nu=None, bc=None, left_bc=None):
        device = data.device
        self.data = data
        self.x = x.to(device)
        self.t_grid = t_grid.to(device)
        self.dx = (dx if dx is not None else (x[1] - x[0])).to(device)
        self.dt = (dt if dt is not None else (t_grid[1] - t_grid[0])).to(device)
        self.nx = nx
        self.nt = nt
        self.rho = rho
        self.nu = nu
        self.bc = bc
        self.left_bc = left_bc

    def ic_residual(self, u_flat):
        u = u_flat.view(self.nx, self.nt)
        ic_target = self.data[0][:, 0].to(u.device)
        return u[:, 0] - ic_target

    def mass_residual_heat(self, u_flat):
        """
        Mass conservation residual for heat. 
        """
        u = u_flat.view(self.nx, self.nt)
        dx = self.dx.to(u.device)
        mass_0 = torch.sum(u[:, 0]) * dx
        mass_t = torch.sum(u, dim=0) * dx
        return mass_t[1:] - mass_0

    def mass_residual_rd(self, u_flat):
        """
        Mass conservation residual for reaction-diffusion. 
        """
        sol = u_flat.view(self.nx, self.nt).T  # [nt, nx]
        device = sol.device
        dx = self.dx.to(device)
        dt = (self.t_grid[1:] - self.t_grid[:-1]).to(device)

        mass = sol.sum(dim=1) * dx
        S = self.rho * (sol * (1 - sol)).sum(dim=1) * dx
        S_mid = 0.5 * (S[:-1] + S[1:])
        S_cum = torch.cat([torch.zeros(1, device=device), torch.cumsum(S_mid * dt, dim=0)], dim=0)

        gL_t = -self.nu * (-25*sol[:, 0] + 48*sol[:, 1] - 36*sol[:, 2] + 16*sol[:, 3] - 3*sol[:, 4]) / (12 * dx)
        gR_t = -self.nu * (25*sol[:, -1] - 48*sol[:, -2] + 36*sol[:, -3] - 16*sol[:, -4] + 3*sol[:, -5]) / (12 * dx)
        F = gL_t - gR_t
        F_mid = 0.5 * (F[:-1] + F[1:])
        F_cum = torch.cat([torch.zeros(1, device=device), torch.cumsum(F_mid * dt, dim=0)], dim=0)

        return mass - (mass[0] + S_cum + F_cum)
        
    def bc_residual_burgers(self, u_flat, start_step=0):
        u = u_flat.view(self.nx, self.nt).T   # [nt,nx]
        # Dirichlet left @ x=0
        resL = u[start_step:, 0] - self.left_bc.to(u.device) 
        # Neumann right zero-gradient @ x=-1
        resR = u[start_step:, -1] - u[start_step:, -2]
        return torch.cat([resL, resR], dim=0)

    def mass_residual_burgers(self, u_flat):
        """
        Mass conservation residual for Burgers equation. 
        """
        u = u_flat.view(self.nx, self.nt).T  # shape: [nt, nx]
        mass = u.sum(dim=1) * self.dx.to(u.device)
        f = 0.5 * u**2
        flux = f[:, -1] - f[:, 0]
        flux_mid = 0.5 * (flux[:-1] + flux[1:])
        flux_cum = torch.cat([
            torch.zeros(1, device=u.device),
            torch.cumsum(flux_mid * self.dt.to(u.device), dim=0)
        ], dim=0)
        return mass - (mass[0] - flux_cum)

    def godunov_flux(self, uL, uR):
        """
        Godunov flux in Burger's equation.
        """
        fL = 0.5 * uL ** 2
        fR = 0.5 * uR ** 2
        s = 0.5 * (uL + uR)
        flux_rarefaction = torch.minimum(fL, fR)
        flux_shock = torch.where(s > 0, fL, fR)
        is_shock = (uL > uR)
        return torch.where(is_shock, flux_shock, flux_rarefaction)

    def burgers_local_multistep_residual(self, u_flat, k=5):
        """
        Local multi-step constraint for Burgers.
        """
        u = u_flat.view(self.nx, self.nt)
        residuals = []
        for n in range(k):
            if n + 1 >= self.nt:
                break
            u_prev = u[:, n]
            u_next = u[:, n + 1]
            uL = u_prev[:-1]
            uR = u_prev[1:]
            F = self.godunov_flux(uL, uR)
            flux_diff = F[1:] - F[:-1]
            rhs = u_prev[1:-1] - (self.dt / self.dx) * flux_diff
            res = u_next[1:-1] - rhs
            residuals.append(res)
        return torch.cat(residuals, dim=0)

    def full_residual_heat(self, u_flat):
        """
        Combined residual for heat.
        """
        return torch.cat([self.ic_residual(u_flat), self.mass_residual_heat(u_flat)], dim=0)

    def full_residual_rd(self, u_flat):
        """
        Combined residual for reaction-diffusion.
        """
        ic = self.ic_residual(u_flat)
        mass = self.mass_residual_rd(u_flat)[1:]
        return torch.cat([ic, mass], dim=0)
    
    def full_residual_burgers(self, u_flat, k=5):
        """
        Combined residual for Burgers IC.
        """
        ic = self.ic_residual(u_flat)
        dyn = self.burgers_local_multistep_residual(u_flat, k=k)
        mass = self.mass_residual_burgers(u_flat)[1:]
        return torch.cat([ic, dyn, mass], dim=0)

    # Burgers BC case 
    def full_residual_burgers2(self, u_flat, start_step=1):
        """
        Combined residual for Burgers BC.
        """
        bc   = self.bc_residual_burgers(u_flat, start_step)
        mass = self.mass_residual_burgers(u_flat)[1:]
        return torch.cat([bc, mass], dim=0)


class Residuals2D:
    def __init__(self, data, x, y, t_grid, dx=None, dy=None, dt=None, nx=None, ny=None, nt=None, rho=None, nu=None):
        device = data.device
        self.data = data
        self.x = x.to(device)
        self.y = y.to(device)
        self.t_grid = t_grid.to(device)
        self.dx = (dx if dx is not None else (x[1] - x[0])).to(device)
        self.dy = (dy if dy is not None else (y[1] - y[0])).to(device)
        self.dt = (dt if dt is not None else (t_grid[1] - t_grid[0])).to(device)
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.rho = rho
        self.nu = nu

    def ic_residual_ns(self, u_flat):
        u = u_flat.view(self.nx, self.ny, self.nt)
        target = self.data[0][:, :, 0].to(u.device)
        return (u[:, :, 0] - target).flatten()

    def mass_residual_ns(self, u_flat):
        """
        Mass conservation residual for Navier-Stokes equation.
        """
        u = u_flat.view(self.nx, self.ny, self.nt)
        dx = self.dx.to(u.device)
        dy = self.dy.to(u.device)
        mass0 = u[:, :, 0].sum() * dx * dy
        mass_t = u.sum(dim=(0, 1)) * dx * dy
        return mass_t[1:] - mass0
    
    def full_residual_ns(self, u_flat):
        """
        Combined residual for Navier-Stokes equation.
        """
        return torch.cat([self.ic_residual_ns(u_flat), self.mass_residual_ns(u_flat)], dim=0)


class Residuals3D:
    """
    Residuals for steady-state 3D Darcy flow: -div(K grad p) = 0 on the unit
    cube, Dirichlet BC on the x=0/x=1 faces, no-flow (Neumann) elsewhere.
    Unlike Residuals/Residuals2D (which enforce an IC + a *global* mass
    conservation law over time), this is a pure spatial BVP: the constraint
    set is a BC residual plus a *local* per-cell finite-volume flux-balance
    residual, evaluated against a fixed, sample-specific permeability field k.
    """
    def __init__(self, k, p_left, p_right, x=None, y=None, z=None, dx=None, dy=None, dz=None, nx=None, ny=None, nz=None):
        device = k.device
        self.k = k
        self.p_left = p_left
        self.p_right = p_right
        self.nx = nx if nx is not None else k.shape[0]
        self.ny = ny if ny is not None else k.shape[1]
        self.nz = nz if nz is not None else k.shape[2]
        self.dx = (dx if dx is not None else (x[1] - x[0])).to(device) if dx is not None or x is not None else None
        self.dy = (dy if dy is not None else (y[1] - y[0])).to(device) if dy is not None or y is not None else None
        self.dz = (dz if dz is not None else (z[1] - z[0])).to(device) if dz is not None or z is not None else None
        if self.dx is None:
            self.dx = self.dy = self.dz = torch.tensor(1.0 / self.nx, device=device)

    def pde_residual_darcy3d(self, p_flat):
        """
        Local finite-volume flux-balance residual: for every cell, the net
        harmonic-averaged flux from its neighbors must vanish, with the
        Dirichlet BC entering as a flux source term at the x=0/x=1 cells
        (standard cell-centered FV treatment -- the boundary *face* value is
        p_left/p_right, not the boundary *cell-center* value, which sits a
        half-cell away). Zero at the true FV solution; alone this fully
        determines p, so it doubles as the full residual for this dataset.
        """
        p = p_flat.view(self.nx, self.ny, self.nz)
        k = self.k.to(p.device)
        h = self.dx.to(p.device)

        def harmonic(k1, k2):
            return 2 * k1 * k2 / (k1 + k2)

        R = torch.zeros_like(p)

        Tx = harmonic(k[:-1], k[1:]) * h
        diff_x = p[1:] - p[:-1]
        R[:-1] = R[:-1] + Tx * diff_x
        R[1:] = R[1:] - Tx * diff_x

        Ty = harmonic(k[:, :-1], k[:, 1:]) * h
        diff_y = p[:, 1:] - p[:, :-1]
        R[:, :-1] = R[:, :-1] + Ty * diff_y
        R[:, 1:] = R[:, 1:] - Ty * diff_y

        Tz = harmonic(k[:, :, :-1], k[:, :, 1:]) * h
        diff_z = p[:, :, 1:] - p[:, :, :-1]
        R[:, :, :-1] = R[:, :, :-1] + Tz * diff_z
        R[:, :, 1:] = R[:, :, 1:] - Tz * diff_z

        Tw = 2 * k[0] * h
        R[0] = R[0] + Tw * (self.p_left - p[0])
        Te = 2 * k[-1] * h
        R[-1] = R[-1] + Te * (self.p_right - p[-1])

        return R.flatten()

    def full_residual_darcy3d(self, p_flat):
        """
        Combined residual for 3D Darcy flow (alias for pde_residual_darcy3d,
        kept for naming consistency with Residuals/Residuals2D's full_residual_*).
        """
        return self.pde_residual_darcy3d(p_flat)
