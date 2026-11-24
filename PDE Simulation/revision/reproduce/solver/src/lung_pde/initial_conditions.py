from dolfin import Expression, Function, UserExpression, assemble, assign, interpolate, project

from .mesh import MUCUS_FLAG


class InitialM1(UserExpression):
    def __init__(self, M0=0.0, **kwargs):
        super().__init__(**kwargs)
        self.M0 = M0

    def eval(self, values, x):
        if x[1] < 0.1:
            values[0] = 0.0
        else:
            values[0] = self.M0

    def value_shape(self):
        return ()


class InitialM2(UserExpression):
    def eval(self, values, x):
        values[0] = 1e-5 * 0

    def value_shape(self):
        return ()


class InitialB1(UserExpression):
    def __init__(self, B0=0.0, **kwargs):
        super().__init__(**kwargs)
        self.B0 = B0

    def eval(self, values, x):
        if x[1] < 0.1:
            values[0] = 0.0
        else:
            values[0] = self.B0

    def value_shape(self):
        return ()


class InitialB2(UserExpression):
    def eval(self, values, x):
        values[0] = 1e-5 * 0

    def value_shape(self):
        return ()


class InitialMucus(UserExpression):
    def eval(self, values, x):
        if x[1] < 0.1:
            values[0] = 0.0
        else:
            values[0] = 1e-3

    def value_shape(self):
        return ()


def initialize_state(bundle, initial_bacteria, initial_macrophages, *, interpolate_profile=False):
    state = Function(bundle.function_space)
    previous = Function(bundle.function_space)
    b_space = bundle.function_space.sub(0).collapse()
    m_space = bundle.function_space.sub(2).collapse()

    profile = interpolate if interpolate_profile else project
    bacteria_profile = profile(
        Expression(
            "x[0]*x[0]*(.2-x[0])*(.2-x[0])*(.2-x[1])*(.2-x[1])",
            degree=2,
        ),
        b_space,
    )
    macrophage_profile = profile(
        Expression(
            "x[0]*x[0]*(.2-x[0])*(.2-x[0])*(.2-x[1])*(.2-x[1])",
            degree=2,
        ),
        m_space,
    )
    profile_norm = assemble(bacteria_profile * bundle.dx(MUCUS_FLAG)) * 50.0

    assign(
        state.sub(0),
        project(
            bacteria_profile
            / profile_norm
            * InitialB1(B0=initial_bacteria, degree=1),
            b_space,
        ),
    )
    assign(
        state.sub(2),
        project(
            macrophage_profile
            / profile_norm
            * InitialM1(M0=initial_macrophages, degree=1),
            m_space,
        ),
    )
    assign(
        state.sub(1),
        interpolate(InitialB2(degree=1), bundle.function_space.sub(1).collapse()),
    )
    assign(
        state.sub(3),
        interpolate(InitialM2(degree=1), bundle.function_space.sub(3).collapse()),
    )
    assign(
        state.sub(4),
        interpolate(
            InitialMucus(degree=1), bundle.function_space.sub(4).collapse()
        ),
    )

    assign(
        previous.sub(0),
        project(
            bacteria_profile
            / profile_norm
            * InitialB1(B0=initial_bacteria, degree=1),
            b_space,
        ),
    )
    assign(
        previous.sub(2),
        project(
            macrophage_profile
            / profile_norm
            * InitialM1(M0=initial_macrophages, degree=1),
            m_space,
        ),
    )
    assign(
        previous.sub(1),
        interpolate(InitialB2(degree=1), bundle.function_space.sub(1).collapse()),
    )
    assign(
        previous.sub(3),
        interpolate(InitialM2(degree=1), bundle.function_space.sub(3).collapse()),
    )
    assign(
        previous.sub(4),
        interpolate(
            InitialMucus(degree=1), bundle.function_space.sub(4).collapse()
        ),
    )
    return state, previous
