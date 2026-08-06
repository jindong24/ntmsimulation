from dolfin import (
    Constant,
    TestFunctions,
    TrialFunction,
    avg,
    derivative,
    dot,
    exp,
    grad,
    inner,
    split,
)

from lung_pde.form_context import FormBundle, FormContext
from lung_pde.mesh import (
    INTERFACE_FLAG,
    LOWER_FLAG,
    MUCUS_FLAG,
    RIGHT_MUCUS_FLAG,
    TISSUE_FLAG,
)


def build_form(context: FormContext) -> FormBundle:
    w = context.w
    w_n = context.w_n
    dt = context.dt
    u = context.velocity
    dx = context.dx
    ds = context.ds
    dS = context.dS
    normal = context.normal
    downnormal = context.down_normal
    eta = context.eta
    danti = context.danti
    MM0 = context.initial_macrophages
    GAMMA1_FLAG = INTERFACE_FLAG
    GAMMA2_FLAG = LOWER_FLAG
    GAMMARU_FLAG = RIGHT_MUCUS_FLAG

    parameters = context.parameters
    DB1 = parameters["DB1"]
    DB2 = parameters["DB2"]
    lambdaB = parameters["lambdaB"]
    KB = parameters["KB"]
    KBi = parameters["KBi"]
    dMb = parameters["dMb"]
    dB = parameters["dB"]
    tauB = parameters["tauB"]
    DM = parameters["DM"]
    delta = parameters["delta"]
    dM = parameters["dM"]
    tauM = parameters["tauM"]
    M0 = parameters["M0"]
    alpha0 = parameters["alpha0"]
    alpha1 = parameters["alpha1"]
    Dm = parameters["Dm"]
    dm = parameters["dm"]
    m0 = parameters["m0"]
    beta0 = parameters["beta0"]
    beta1 = parameters["beta1"]

    B1, B2, M1, M2, m = split(w)
    B1_n, B2_n, M1_n, M2_n, m_n = split(w_n)
    E1, E2, F1, F2, s = TestFunctions(w.function_space())
    Tupplus = (inner(normal('+'),downnormal)+1)/2
    Tdownplus = (inner(normal('+'),-downnormal)+1)/2
    Tupminus = (inner(normal('-'),downnormal)+1)/2
    Tdownminus = (inner(normal('-'),-downnormal)+1)/2
    alphaB = alpha0 + alpha1*B2
    alpha = Constant(.01)

    a_B1 = (B1-B1_n)/dt*E1*dx(MUCUS_FLAG) - 1/eta*exp(1*(1.-(m/m0)))*dot(u,grad(E1))*B1*dx(MUCUS_FLAG) + 1/eta*exp(1*(1.-(m/m0)))*DB1*inner(grad(B1),grad(E1))*dx(MUCUS_FLAG) \
                - (lambdaB*B1*(1-B1/KB)+dMb*B1/(B1+KBi)*M1)*E1*dx(MUCUS_FLAG) \
                - 1/eta*exp(1*(1.-(avg(m)/m0)))*DB1*tauB*(avg(B2)-avg(B1))*avg(E1)*dS(GAMMA1_FLAG) + alpha*B1*E1*ds(GAMMARU_FLAG) + danti*MM0*dB*B1*E1*dx(MUCUS_FLAG)
    a_B2 = (B2-B2_n)/dt*E2*dx(TISSUE_FLAG) + 1/eta*exp(1*(1.-(m/m0)))*DB2*inner(grad(B2),grad(E2))*dx(TISSUE_FLAG) \
                - (lambdaB*B2*(1-B2/KB)+dMb*B2/(B2+KBi)*M2)*E2*dx(TISSUE_FLAG) + dB*M2*B2*E2*dx(TISSUE_FLAG)\
                + 1/eta*exp(1*(1.-(avg(m)/m0)))*DB2*tauB*(avg(B2)-avg(B1))*avg(E2)*dS(GAMMA1_FLAG) + danti*MM0*dB*B2*E2*dx(TISSUE_FLAG)
    a_M1 = (M1-M1_n)/dt*F1*dx(MUCUS_FLAG) - 1/eta*exp(1*(1.-(m/m0)))*dot(u,grad(F1))*M1*dx(MUCUS_FLAG) + 1/eta*exp(1*(1.-(m/m0)))*DM*inner(grad(M1),grad(F1))*dx(MUCUS_FLAG) - delta*M1*inner(grad(B1),grad(F1))*dx(MUCUS_FLAG) \
            + (dM*M1+dMb*B1/(B1+KBi)*M1)*F1*dx(MUCUS_FLAG) \
            - 1/eta*exp(1*(1.-(avg(m)/m0)))*DM*tauM*(avg(M2)-avg(M1))*avg(F1)*dS(GAMMA1_FLAG) \
            + delta*(inner(grad(B1('+')),normal('+'))*Tupplus+inner(grad(B1('-')),normal('-'))*Tupminus)*avg(M1)*avg(F1)*dS(GAMMA1_FLAG) + alpha*M1*F1*ds(GAMMARU_FLAG)
    a_M2 = (M2-M2_n)/dt*F2*dx(TISSUE_FLAG) + 1/eta*exp(1*(1.-(m/m0)))*DM*inner(grad(M2),grad(F2))*dx(TISSUE_FLAG) - delta*M2*inner(grad(B2),grad(F2))*dx(TISSUE_FLAG) \
            + (dM*M2+dMb*B2/(B2+KBi)*M2)*F2*dx(TISSUE_FLAG) \
            + 1/eta*exp(1*(1.-(avg(m)/m0)))*DM*tauM*(avg(M2)-avg(M1))*avg(F2)*dS(GAMMA1_FLAG) \
            + delta*(inner(grad(B2('+')),normal('+'))*Tdownplus+inner(grad(B2('-')),normal('-'))*Tdownminus)*avg(M2)*avg(F2)*dS(GAMMA1_FLAG) \
            + alphaB*(M2-M0)*F2*ds(GAMMA2_FLAG)
    a_m = (m-m_n)/dt*s*dx(MUCUS_FLAG) - 1/eta*exp(1*(1.-(m/m0)))*dot(u,grad(s))*m*dx(MUCUS_FLAG) + Dm*inner(grad(m),grad(s))*dx(MUCUS_FLAG) \
            + dm*m*s*dx(MUCUS_FLAG) \
            + (beta0+beta1*avg(M1)*avg(B1))*(avg(m)-m0)*avg(s)*dS(GAMMA1_FLAG) + alpha*m*s*ds(GAMMARU_FLAG)
    a = a_B1 + a_B2 + a_M1 + a_M2 + a_m

    J = derivative(a, w, TrialFunction(w.function_space()))
    return FormBundle(a, J, (a_B1, a_B2, a_M1, a_M2, a_m))
