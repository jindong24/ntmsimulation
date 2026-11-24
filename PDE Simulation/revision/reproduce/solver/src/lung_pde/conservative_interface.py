from dolfin import TestFunctions, TrialFunction, avg, derivative, exp, split
from ufl import Form

from .form_context import FormBundle
from .mesh import INTERFACE_FLAG


def interface_fluxes(w, parameters, eta=None, *, macrophage_diffusivity_in_flux=True):
    B1, B2, M1, M2, mucus = split(w)
    # Positive flux is out of the mucus compartment.
    j_b = parameters["tauB"] * (avg(B1) - avg(B2))
    tau_m = parameters["tauM"]
    if macrophage_diffusivity_in_flux:
        tau_m = parameters["DM"] * tau_m
    j_m = tau_m * (avg(M1) - avg(M2))
    if eta is not None:
        mobility = exp(1.0 - avg(mucus) / parameters["m0"]) / eta
        j_b *= mobility
        j_m *= mobility
    return j_b, j_m


def replace_interface_exchange(legacy: FormBundle, context, *, viscosity_scaled=False,
                               macrophage_diffusivity_in_flux=True) -> FormBundle:
    """Keep the volume and outer-boundary terms, and replace exchange on Gamma 1."""
    w = context.w
    E1, E2, F1, F2, _ = TestFunctions(w.function_space())
    measure = context.dS(INTERFACE_FLAG)
    j_b, j_m = interface_fluxes(
        w, context.parameters, context.eta if viscosity_scaled else None,
        macrophage_diffusivity_in_flux=macrophage_diffusivity_in_flux,
    )
    fluxes = (j_b * avg(E1), -j_b * avg(E2), j_m * avg(F1), -j_m * avg(F2))
    bulk_and_outer = (
        Form([
            integral for integral in term.integrals()
            if not (
                integral.integral_type() == "interior_facet"
                and integral.subdomain_id() == INTERFACE_FLAG
            )
        ])
        for term in legacy.terms[:4]
    )
    terms = tuple(
        term + flux * measure for term, flux in zip(bulk_and_outer, fluxes)
    ) + (legacy.terms[4],)
    residual = sum(terms)
    jacobian = derivative(residual, w, TrialFunction(w.function_space()))
    return FormBundle(residual, jacobian, terms)
