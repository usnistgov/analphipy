import numpy as np

# classes to handle pair potentials


class _Phidphi_class:
    def __call__(self, r, dphi=False):
        if dphi:
            out = self.phidphi(r)
        else:
            out = self.phi(r)
        return out

    # @property
    # def segments(self):
    #     raise NotImplementedError('to be implemented in subclass')

    def phi(self, r):
        NotImplementedError("to be implemented in subclass if appropriate")

    def phidphi(self, r):
        NotImplementedError("to be implemented in subclass if appropriate")


class Phi_cut_base(_Phidphi_class):
    """
    create cut potential from base potential


    phi_cut(r) = phi(r) + vcorrect(r) if r <= rcut
               = 0 otherwise


    dphi_cut(r) = dphi(r) + dvcorrect(r) if r <= rcut
                = 0 otherwise

    So, for example, for just cut, `vcorrect(r) = -v(rcut)`, `dvrcut=0`,
    and for lfs, `vcorrect(r) = -v(rcut) - dv(rcut)/dr (r - rcut)`
    `dvcorrect(r) = ...`
    """

    # phi_base : _Phidphi_class
    # rcut : float

    def __init__(self, phi_base, rcut):
        self.phi_base = phi_base
        self.rcut = rcut

    @property
    def segments(self):
        return [x for x in self.phi_base.segments if x < self.rcut] + [self.rcut]

    @classmethod
    def from_base(cls, base, rcut, *args, **kws):
        """
        Create cut potential from base phi class
        """
        phi_base = base(*args, **kws)
        return cls(phi_base=phi_base, rcut=rcut)

    def phi(self, r):
        r = np.asarray(r)
        v = np.empty_like(r)

        left = r <= self.rcut
        right = ~left

        v[right] = 0.0

        if any(left):
            v[left] = self.phi_base(r[left]) + self.vcorrect(r[left])

        return v

    def phidphi(self, r):
        r = np.array(r)

        v = np.empty_like(r)
        dv = np.empty_like(r)

        left = r <= self.rcut
        right = ~left

        v[right] = 0.0
        dv[right] = 0.0

        if np.any(left):
            v[left], dv[left] = self.phi_base.phidphi(r[left])
            v[left] += self.vcorrect(r[left])
            dv[left] += self.dvcorrect(r[left])

        return v, dv

    def vcorrect(self, r):
        """ """
        raise NotImplementedError

    def dvcorrect(self, r):
        raise NotImplementedError

    def __repr__(self):
        name = type(self).__name__
        params = "rcut={rcut}, phi_base={phi_base}".format(
            rcut=self.rcut, phi_base=repr(self.phi_base)
        )
        return f"{name}({params})"


class Phi_cut(Phi_cut_base):
    """
    potential phi cut at position r
    """

    def __init__(self, phi_base, rcut):
        super().__init__(phi_base, rcut)
        self.vcut = self.phi_base(self.rcut)

    def vcorrect(self, r):
        return -self.vcut

    def dvcorrect(self, r):
        return 0.0


class Phi_lfs(Phi_cut_base):
    def __init__(self, phi_base, rcut):
        super().__init__(phi_base, rcut)
        vcut, dvcut = self.phi_base.phidphi(rcut)
        self.vcut = vcut
        self.dvdrcut = -dvcut * rcut

    def vcorrect(self, r):
        return -(self.vcut + self.dvdrcut * (r - self.rcut))

    def dvcorrect(self, r):
        return self.dvdrcut / r


class Phi_base(_Phidphi_class):
    def __init__(self, segments, **kws):
        self.segments = segments
        self.params = kws

    def cut(self, rcut):
        return Phi_cut(phi_base=self, rcut=rcut)

    def lfs(self, rcut):
        return Phi_lfs(phi_base=self, rcut=rcut)

    def copy(self):
        return type(self)(**self.params)

    def phi(self, r):
        raise NotImplementedError("to be implemented in subclass")

    def phidphi(self, r):
        raise NotImplementedError("to be implemented in subclass or not available")

    def __repr__(self):
        name = type(self).__name__
        params = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{name}({params})"


class Phi_lj(Phi_base):
    def __init__(self, sig=1.0, eps=1.0):
        super().__init__(segments=[0.0, np.inf], sig=sig, eps=eps)
        self.sigsq = sig * sig
        self.four_eps = 4.0 * eps

    def phi(self, r):
        r = np.array(r)
        x2 = self.sigsq / (r * r)
        x6 = x2 ** 3
        return self.four_eps * x6 * (x6 - 1.0)

    def phidphi(self, r):
        """calculate phi and dphi (=-1/r dphi/dr) at particular r"""
        r = np.array(r)
        rinvsq = 1.0 / (r * r)

        x2 = self.sigsq * rinvsq
        x6 = x2 * x2 * x2

        phi = self.four_eps * x6 * (x6 - 1.0)
        dphi = 12.0 * self.four_eps * x6 * (x6 - 0.5) * rinvsq
        return phi, dphi


class Phi_nm(Phi_base):
    def __init__(self, n=12, m=6, sig=1.0, eps=1.0):
        super().__init__(segments=[0.0, np.inf], n=n, m=m, sig=sig, eps=eps)
        self.sig = sig
        self.n = n
        self.m = m
        self.prefac = eps * (n / (n - m)) * (n / m) ** (m / (n - m))

    def phi(self, r):

        r = np.array(r)

        x = self.sig / r
        return self.prefac * (x ** self.n - x ** self.m)

    def phidphi(self, r):

        r = np.array(r)

        x = self.sig / r

        xn = x ** self.n
        xm = x ** self.m
        phi = self.prefac * (xn - xm)

        # dphi = -1/r dphi/dr = x dphi/dx * 1/r**2
        # where x = sig / r
        dphi = self.prefac * (self.n * xn - self.m * xm) / (r ** 2)

        return phi, dphi


class CubicTable(Phi_base):
    def __init__(self, bounds, phi_array):
        super().__init__(segments=list(bounds), bounds=bounds, phi_array=phi_array)

        self.phi_array = np.asarray(phi_array)
        self.bounds = bounds

        self.ds = (self.bounds[1] - self.bounds[0]) / self.size
        self.dsinv = 1.0 / self.ds

    @classmethod
    def from_phi(cls, phi, rmin, rmax, ds):
        bounds = (rmin * rmin, rmax * rmax)

        delta = bounds[1] - bounds[0]

        size = int(delta / ds)
        ds = delta / size

        phi_array = []

        r = np.sqrt(np.arange(size + 1) * ds + bounds[0])

        phi_array = phi(r)

        return cls(bounds=bounds, phi_array=phi_array)

    def __len__(self):
        return len(self.phi_array)

    @property
    def size(self):
        return len(self) - 1

    @property
    def smin(self):
        return self.bounds[0]

    @property
    def smax(self):
        return self.bounds[1]

    def phidphi(self, r):
        r = np.asarray(r)

        v = np.empty_like(r)
        dv = np.empty_like(r)

        s = r * r
        left = s <= self.smax
        right = ~left

        v[right] = 0.0
        dv[right] = 0.0

        if np.any(left):

            sds = (s[left] - self.smin) * self.dsinv
            k = sds.astype(int)
            k[k < 0] = 0

            xi = sds - k

            t = np.take(self.phi_array, [k, k + 1, k + 2], mode="clip")
            dt = np.diff(t, axis=0)
            ddt = np.diff(dt, axis=0)

            v[left] = t[0, :] + xi * (dt[0, :] + 0.5 * (xi - 1.0) * ddt[0, :])
            dv[left] = -2.0 * self.dsinv * (dt[0, :] + (xi - 0.5) * ddt[0, :])

        return v, dv

    def phi(self, r):
        return self.phidphi(r)[0]


class Phi_yk(Phi_base):
    def __init__(self, z=1.0, sig=1.0, eps=1.0):
        super().__init__(segments=[0.0, sig, np.inf], z=z, sig=sig, eps=eps)
        self.sig = sig
        self.eps = eps
        self.z = z

    def phi(self, r):

        sig, eps = self.sig, self.eps

        r = np.array(r)
        phi = np.empty_like(r)
        m = r >= sig

        phi[~m] = np.inf
        if np.any(m):
            x = r[m] / sig
            phi[m] = -eps * np.exp(-self.z * (x - 1.0)) / x
        return phi


class Phi_hs(Phi_base):
    def __init__(self, sig=1.0):
        super().__init__(segments=[0.0, sig], sig=1.0)

        self.sig = sig

    def phi(self, r):
        r = np.array(r)

        phi = np.empty_like(r)

        m0 = r < self.sig
        phi[m0] = np.inf
        phi[~m0] = 0.0
        return phi


class Phi_sw(Phi_base):
    def __init__(self, sig=1, eps=-1.0, lam=1.5):
        super().__init__(segments=[0.0, sig, sig * lam], sig=sig, eps=eps, lam=lam)
        self.sig, self.eps, self.lam = sig, eps, lam

    def phi(self, r):

        sig, eps, lam = self.sig, self.eps, self.lam

        r = np.array(r)

        phi = np.empty_like(r)

        m0 = r < sig
        m2 = r >= lam * sig
        m1 = (~m0) & (~m2)

        phi[m0] = np.inf
        phi[m1] = eps
        phi[m2] = 0.0

        return phi


def phi_yk(r, z, sig=1.0, eps=1.0):
    r = np.array(r)
    phi = np.empty_like(r)
    m = r >= sig

    phi[~m] = np.inf
    if np.any(m):
        x = r[m] / sig
        phi[m] = -eps * np.exp(-z * (x - 1.0)) / x
    return phi


def phi_sw(r, sig=1.0, eps=-1.0, lam=1.5):
    """
    Square well potential with value

    * inf : r < sig
    * eps : sig <= r < lam * sig
    * 0 :  r >= lam * sig
    """

    r = np.array(r)

    phi = np.empty_like(r)

    m0 = r < sig
    m2 = r >= lam * sig
    m1 = (~m0) & (~m2)

    phi[m0] = np.inf
    phi[m1] = eps
    phi[m2] = 0.0

    return phi


def phi_hs(r, sig=1.0):

    r = np.array(r)

    phi = np.empty_like(r)

    m0 = r < sig
    phi[m0] = np.inf
    phi[~m0] = 0.0

    return phi
