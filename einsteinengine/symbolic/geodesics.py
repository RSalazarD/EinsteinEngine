import sympy as sp
import symengine as se

class Geodesics:
    r"""
    Computes the geodesic equations of motion for a given space-time.
    d^2 x^\mu / d\tau^2 = - \Gamma^\mu_{\alpha\beta} (dx^\alpha / d\tau) (dx^\beta / d\tau)
    """
    
    def __init__(self, christoffel, param_name="tau", verbose=False):
        self.christoffel = christoffel
        self.syms = christoffel.syms
        self.dims = christoffel.dims
        self.verbose = verbose
        
        # --- THEORETICAL VARIABLES (SymPy) ---
        # Used ONLY for LaTeX display. SymEngine doesn't support abstract uncomputed Functions well.
        self.tau_sp = sp.Symbol(param_name, real=True)
        self.x_funcs_sp = [sp.Function(s.name)(self.tau_sp) for s in self.syms]
        
        # --- PERFORMANCE VARIABLES (SymEngine / C++) ---
        # Algebraic velocities used for high-speed numerical simulations
        self.v_syms_se = [se.Symbol(f"v_{s.name}") for s in self.syms]
        
        if self.verbose:
            print(f"Initialized Geodesics equations generator. Affine parameter: {self.tau_sp}")

    def get_equations(self, substitute_velocities=True, simplify=False):
        """
        Generates the right-hand side (accelerations) of the geodesic equations.
        
        Args:
            substitute_velocities (bool): 
                - If True (Default): C++ HIGH PERFORMANCE MODE. Returns algebraic system for SciPy.
                - If False: PYTHON THEORETICAL MODE. Returns pure differential equations.
            simplify (bool): If True, attempts to simplify the final expressions (can be slow).
        """
        if self.verbose:
            print("Constructing the geodesic differential equations...")
            
        Gamma = self.christoffel.get_raw_data()
        accelerations = []
        
        for mu in range(self.dims):
            # Accumulator initialized purely in C++
            accel_mu = se.sympify(0) if substitute_velocities else sp.sympify(0)
            
            for alpha in range(self.dims):
                for beta in range(self.dims):
                    # Fetch the Christoffel symbol component
                    gamma_se = se.sympify(Gamma[mu][alpha][beta])
                    
                    if gamma_se == 0:
                        continue
                        
                    if substitute_velocities:
                        # 1. NUMERICAL MODE (SymEngine C++)
                        # Pure, blazing fast algebra
                        vel_alpha = self.v_syms_se[alpha]
                        vel_beta = self.v_syms_se[beta]
                        accel_mu += - gamma_se * vel_alpha * vel_beta
                    else:
                        # 2. THEORETICAL MODE (SymPy fallback)
                        # We must cross the bridge to Python to use sp.Derivative and sp.Function
                        gamma_sp = sp.sympify(gamma_se)
                        subs_dict = {self.syms[i]: self.x_funcs_sp[i] for i in range(self.dims)}
                        gamma_sp = gamma_sp.subs(subs_dict)
                        
                        vel_alpha = sp.Derivative(self.x_funcs_sp[alpha], self.tau_sp)
                        vel_beta = sp.Derivative(self.x_funcs_sp[beta], self.tau_sp)
                        accel_mu += - gamma_sp * vel_alpha * vel_beta
            
            # Performance delegation for simplification
            if simplify:
                if self.verbose: print(f"Simplifying equation for coordinate {self.syms[mu]}...")
                # 1. C++ -> Python (sp.sympify)
                cleaned_python_eq = sp.simplify(sp.sympify(accel_mu))
                # 3. Python -> C++ (se.sympify)
                accelerations.append(se.sympify(cleaned_python_eq))
            else:
                accelerations.append(accel_mu)
            
        return accelerations

    def display_equations(self):
        """
        Helper method to render the theoretical differential equations in Jupyter Notebooks.
        """
        from IPython.display import display, Math
        
        # Force theoretical mode and skip simplification for instant rendering
        eqs = self.get_equations(substitute_velocities=False, simplify=False)
        for i, eq in enumerate(eqs):
            coord = self.syms[i].name
            latex_str = f"\\frac{{d^2 {coord}}}{{d\\tau^2}} = {sp.latex(eq)}"
            display(Math(latex_str))