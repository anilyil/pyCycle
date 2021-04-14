import openmdao.api as om
import pycycle.api as pyc


class PoddedFan(pyc.Cycle):
    # def initialize(self):
        # self.options.declare("design", default=True, desc="Switch between on-design and off-design calculation.")
        # super().initialize()

    def setup(self):

        design = self.options["design"]

        params = self.add_subsystem("params", om.IndepVarComp(), promotes=["*"])  # constant values that never change

        # this is just set here to provide a good value for this quantity
        # this will not affect the stuff we are interested in
        params.add_output("fan:MN_out_des", 0.62)
        if design:
            self.connect("fan:MN_out_des", "fan.MN")

        # compute total quantities for prop. model using static quantities in CFD
        self.add_subsystem(
            "cfd_start",
            pyc.CFDStart(),
            promotes_inputs=["Ps"],
        )

        self.add_subsystem(
            "fan",
            pyc.Compressor(
                map_data=pyc.FanMap,
                design=design,
                # thermo_data=thermo_spec,
                # elements=AIR_MIX,
                # bleed_names=[],
                map_extrap=True,
            ),
            promotes_inputs=["Nmech", ("PR", "PRdes")],
        )

        # we match eta_a to 0.97 for the design case
        if design:
            balance = self.add_subsystem("balance", om.BalanceComp())
            balance.add_balance("fan_eta_a", lower=0.7, upper=0.999, rhs_val=0.97)
            self.connect("balance.fan_eta_a", "fan.eff")
            self.connect("fan.eff_poly", "balance.lhs:fan_eta_a")

        # we match the FPR using nmech for off-design
        else:
            # add balance
            balance = self.add_subsystem("balance", om.BalanceComp())

            # # vary nmech to match power
            # balance.add_balance('Nmech', val=600., units='rpm', lower=0.1, upper=1200.0, eq_units='hp')
            # self.connect('fan.power', 'balance.lhs:Nmech')

            # vary nmech to match FPR
            balance.add_balance("Nmech", val=600.0, units="rpm", lower=200.0, upper=1200.0, eq_units=None)
            self.connect("fan.PR", "balance.lhs:Nmech")
            self.connect("balance.Nmech", "Nmech")

        self.add_subsystem(
            "amb",
            pyc.FlightConditions(),
            promotes_inputs=[("MN", "mach"), ("alt", "altitude")],
        )

        # convert the unit and sign here
        self.add_subsystem(
            "perf",
            om.ExecComp(
                [
                    "NPR = pt2/p_amb",
                    "shaft_pwr  = -fan_power",
                    "delta_heat = mdot * (h_real - h_ideal)",
                ],
                pt2={"units": "psi"},
                p_amb={"units": "psi"},
                fan_power={"units": "kW"},
                shaft_pwr={"units": "kW"},
                delta_heat={"units": "kW", "value": 0.0},
                mdot={"units": "kg/s"},
                h_real={"units": "kJ/kg"},
                h_ideal={"units": "kJ/kg"},
            ),
            promotes_outputs=["NPR", "shaft_pwr", "delta_heat"],
        )

        self.connect("amb.Fl_O:stat:P", "perf.p_amb")
        self.connect("fan.Fl_O:tot:P", "perf.pt2")
        self.connect("fan.power", "perf.fan_power")

        # connections for delta_heat computation
        # mdot is connected on the upper level
        self.connect("fan.enth_rise.ht_out", "perf.h_real")
        self.connect("fan.ideal_flow.h", "perf.h_ideal")

        self.pyc_connect_flow("cfd_start.Fl_O", "fan.Fl_I")

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-10
        newton.options["rtol"] = 1e-10
        newton.options["iprint"] = 2
        self.set_solver_print(level=-1)
        self.set_solver_print(level=2, depth=3)
        newton.options["maxiter"] = 15
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 10
        # newton.options['debug_print'] = True
        # newton.options['err_on_maxiter'] = True

        # off-design needs a bit more robustness
        if not design:
            newton.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar")

        self.linear_solver = om.DirectSolver()

        # base_class setup should be called as the last thing in your setup
        super().setup()


class PropulsionGroup(om.Group):
    def initialize(self):
        self.options.declare("design", default=True)
        self.options.declare("MNGuess", default=0.6)
        self.options.declare("mnrho", default=100.0)
        self.options.declare("fan_model", default="az")

    def setup(self):
        design = self.options["design"]
        MNGuess = self.options["MNGuess"]
        mnrho = self.options["mnrho"]
        fan_model = self.options["fan_model"]

        # compute FPR
        self.add_subsystem(
            "FPR", om.ExecComp("PRdes = Pt2/Pt1", Pt1={"units": "Pa"}, Pt2={"units": "Pa"}), promotes=["*"]
        )

        # convert half-body integrated quantities to full-body
        self.add_subsystem(
            "full_body",
            om.ExecComp(
                [
                    "W_full = 2*W",
                    "area_full = 2*area",
                    "V=Vx*cos(alpha) + Vy*sin(alpha)",
                    "area_exit_full = 2*area_exit",
                    "adflow_pwr = 2 * adflow_fan_pwr",
                ],
                area_full={"units": "inch**2"},
                area={"units": "inch**2"},
                area_exit_full={"units": "inch**2"},
                area_exit={"units": "inch**2"},
                W_full={"units": "lbm/s"},
                W={"units": "lbm/s"},
                V={"units": "ft/s"},
                Vx={"units": "ft/s"},
                Vy={"units": "ft/s"},
                alpha={"units": "rad"},
                adflow_pwr={"units": "kW"},
                adflow_fan_pwr={"units": "kW"},
            ),
            promotes_inputs=["area", "area_exit", "W", "Vx", "Vy", "alpha", "adflow_fan_pwr"],
            promotes_outputs=["adflow_pwr"],
        )

        self.add_subsystem("podded_fan", PoddedFan(design=design), promotes=["*"])

        self.connect("full_body.W_full", "cfd_start.W")
        self.connect("full_body.area_full", "cfd_start.area")
        self.connect("full_body.V", "cfd_start.V")
        self.connect("full_body.W_full", ["amb.W", "perf.mdot"])

        if not design:
            # connect the fan exit area to get the static quantities
            self.connect("full_body.area_exit_full", "fan.area")


if __name__ == "__main__":

    from pycycle.viewers import print_flow_station

    # from pycycle.viewers import print_compressor

    p = om.Problem()

    params = p.model.add_subsystem("params", om.IndepVarComp(), promotes=["*"])
    p.model.add_subsystem("fan", PropulsionGroup(mnrho=100.0))

    # integrated outputs from a CFD model
    params.add_output("Ps", units="Pa", val=31482.90892288)
    params.add_output("Pt1", units="Pa", val=37577.40009854)
    params.add_output("Pt2", units="Pa", val=45900.480453)
    params.add_output("area", units="m**2", val=1.29225432)
    params.add_output("area_exit", units="m**2", val=1.29225432)
    params.add_output("W", units="kg/s", val=94.10411523)
    params.add_output("Vx", units="m/s", val=157.10906576)
    params.add_output("Vy", units="m/s", val=-0.00026193)
    params.add_output("PRdes", val=1.22149164)

    p.model.connect("Ps", "fan.Ps")
    p.model.connect("Pt1", "fan.Pt1")
    p.model.connect("Pt2", "fan.Pt2")
    p.model.connect("area", "fan.area")
    p.model.connect("area_exit", "fan.area_exit")
    p.model.connect("W", "fan.W")
    p.model.connect("Vx", "fan.Vx")
    p.model.connect("Vy", "fan.Vy")

    # flight conditions
    params.add_output("mach", val=0.83)
    params.add_output("altitude", units="m", val=10668.0)
    params.add_output("alpha", units="deg", val=0.0)

    p.model.connect("altitude", "fan.altitude")
    p.model.connect("alpha", "fan.alpha")
    p.model.connect("mach", "fan.mach")

    # solver setup
    p.set_solver_print(level=-1)
    p.set_solver_print(level=2, depth=3)

    p.setup()
    om.n2(p, show_browser=False, outfile="fan.html")

    # run the model...
    p.run_model()

    print("\n\nDESIGN\n\n")
    print_flow_station(p, ["fan.amb.Fl_O", "fan.cfd_start.Fl_O", "fan.fan.Fl_O"])

    # print_compressor(p, ['fan.fan'])

    print("NPR", p["fan.NPR"])
    print("fan_adiabatic_eff", p["fan.fan.eff"])
    print("delta_heat", p["fan.delta_heat"])
    print("shaft power", p["fan.shaft_pwr"])

    p.model.list_inputs(units=True)
    p.model.list_outputs(units=True)
    # p.model.list_outputs(units=True, residuals_tol=1e-3)
