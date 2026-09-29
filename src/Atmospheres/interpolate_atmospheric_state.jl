using Oceananigans.Operators: intrinsic_vector
using Oceananigans.Grids: _node
using Oceananigans.Fields: FractionalIndices, interpolate
using Oceananigans.OutputReaders: cpu_interpolating_time_indices

using ..Oceans: forcing_barotropic_potential

"""Interpolate the atmospheric state onto the ocean / sea-ice grid."""
function EarthSystemModels.interpolate_state!(exchanger, grid, atmosphere::PrescribedAtmosphere, coupled_model)
    atmosphere_grid = atmosphere.grid

    # Basic model properties
    arch = architecture(grid)
    clock = coupled_model.clock

    #####
    ##### First interpolate atmosphere time series
    ##### in time and to the ocean grid.
    #####

    # We use .data here to save parameter space (unlike Field, adapt_structure for
    # fts = FieldTimeSeries does not return fts.data)
    atmosphere_velocities = (u = atmosphere.velocities.u.data,
                             v = atmosphere.velocities.v.data)

    atmosphere_tracers = (T = atmosphere.temperature.data,
                          q = atmosphere.specific_humidity.data)

    precipitation_flux = atmosphere.precipitation_flux
    rain_fts = isnothing(precipitation_flux) ? nothing : precipitation_flux.rain
    snow_fts = isnothing(precipitation_flux) ? nothing : precipitation_flux.snow

    rainfall_flux = surface_rainfall_flux(atmosphere)
    snowfall_flux = surface_snowfall_flux(atmosphere)
    atmosphere_pressure = atmosphere.pressure.data

    atmosphere_fields = exchanger.state
    space_fractional_indices = exchanger.regridder

    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/NumericalEarth.jl/issues/116.
    atmosphere_data = (u   = atmosphere_fields.u.data,
                       v   = atmosphere_fields.v.data,
                       T   = atmosphere_fields.T.data,
                       p   = atmosphere_fields.p.data,
                       q   = atmosphere_fields.q.data,
                       Jʳⁿ = atmosphere_fields.Jʳⁿ.data,
                       Jˢⁿ = atmosphere_fields.Jˢⁿ.data)

    kernel_parameters = interface_kernel_parameters(grid)

    # Each field is interpolated in time on its own time axis: a field whose reload
    # window differs from u's (e.g. ERA5 precipitation, window-centred half an hour off
    # the other instantaneous fields) must not be read using u's time index (issue
    # NumericalEarth.jl#711 — ERA5's total_precipitation is stamped half an hour off
    # u/v/T/q, so reusing u's reload schedule reads precipitation out of its own window).
    t = clock.time
    u_time_args    = time_arguments(arch, atmosphere.velocities.u,   t)
    v_time_args    = time_arguments(arch, atmosphere.velocities.v,   t)
    T_time_args    = time_arguments(arch, atmosphere.temperature,    t)
    q_time_args    = time_arguments(arch, atmosphere.specific_humidity, t)
    p_time_args    = time_arguments(arch, atmosphere.pressure,      t)
    rain_time_args = time_arguments(arch, rain_fts,                 t)
    snow_time_args = time_arguments(arch, snow_fts,                 t)

    launch!(arch, grid, kernel_parameters,
            _interpolate_primary_atmospheric_state!,
            atmosphere_data,
            space_fractional_indices,
            grid,
            atmosphere_velocities,
            u_time_args,
            v_time_args,
            atmosphere_tracers,
            T_time_args,
            q_time_args,
            atmosphere_pressure,
            p_time_args,
            rainfall_flux,
            rain_time_args,
            snowfall_flux,
            snow_time_args)

    # Set ocean barotropic pressure forcing
    #
    # TODO: find a better design for this that doesn't have redundant
    # arrays for the barotropic potential
    potential = forcing_barotropic_potential(coupled_model.ocean)
    ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density

    if !isnothing(potential)
        parent(potential) .= parent(atmosphere_data.p) ./ ρᵒᶜ
    end
end

@inline get_fractional_index(i, j, ::Nothing) = nothing
@inline get_fractional_index(i, j, frac) = @inbounds frac[i, j, 1]

# Nothing-safe: a field the atmosphere doesn't carry (e.g. no precipitation_flux) has no
# time axis to interpolate on; interp_atmos_time_series short-circuits on the ::Nothing data.
time_arguments(arch, ::Nothing, t) = (nothing, nothing, nothing)
time_arguments(arch, fts, t) = (cpu_interpolating_time_indices(arch, fts.times, fts.time_indexing, t), fts.backend, fts.time_indexing)

@kernel function _interpolate_primary_atmospheric_state!(surface_atmos_state,
                                                         space_fractional_indices,
                                                         exchange_grid,
                                                         atmos_velocities,
                                                         u_time_args,
                                                         v_time_args,
                                                         atmos_tracers,
                                                         T_time_args,
                                                         q_time_args,
                                                         atmos_pressure,
                                                         p_time_args,
                                                         rainfall_flux,
                                                         rain_time_args,
                                                         snowfall_flux,
                                                         snow_time_args)

    i, j = @index(Global, NTuple)

    ii = space_fractional_indices.i
    jj = space_fractional_indices.j
    fi = get_fractional_index(i, j, ii)
    fj = get_fractional_index(i, j, jj)

    x_itp = FractionalIndices(fi, fj, nothing)

    uᵃᵗ = interp_atmos_time_series(atmos_velocities.u, x_itp, u_time_args...)
    vᵃᵗ = interp_atmos_time_series(atmos_velocities.v, x_itp, v_time_args...)
    Tᵃᵗ = interp_atmos_time_series(atmos_tracers.T,    x_itp, T_time_args...)
    qᵃᵗ = interp_atmos_time_series(atmos_tracers.q,    x_itp, q_time_args...)
    pᵃᵗ = interp_atmos_time_series(atmos_pressure,     x_itp, p_time_args...)

    Mr = interp_atmos_time_series(rainfall_flux, x_itp, rain_time_args...)
    Ms = interp_atmos_time_series(snowfall_flux, x_itp, snow_time_args...)

    # Convert atmosphere velocities (usually defined on a latitude-longitude grid) to
    # the frame of reference of the native grid
    kᴺ = size(exchange_grid, 3) # index of the top ocean cell
    uᵃᵗ, vᵃᵗ = intrinsic_vector(i, j, kᴺ, exchange_grid, uᵃᵗ, vᵃᵗ)

    @inbounds begin
        surface_atmos_state.u[i, j, 1] = uᵃᵗ
        surface_atmos_state.v[i, j, 1] = vᵃᵗ
        surface_atmos_state.T[i, j, 1] = Tᵃᵗ
        surface_atmos_state.p[i, j, 1] = pᵃᵗ
        surface_atmos_state.q[i, j, 1] = qᵃᵗ
        surface_atmos_state.Jʳⁿ[i, j, 1] = Mr
        surface_atmos_state.Jˢⁿ[i, j, 1] = Ms
    end
end

#####
##### Utility for interpolating tuples of fields
#####

@inline interp_atmos_time_series(::Nothing, X, time, grid, args...) = 0

# Note: assumes loc = (c, c, nothing) (and the third location should not matter.)
@inline interp_atmos_time_series(J::AbstractArray, X::FractionalIndices, time, args...) =
    interpolate(X, time, J, args...)

@inline interp_atmos_time_series(J::AbstractArray, X, time, grid, args...) =
    interpolate(X, time, J, (Center(), Center(), nothing), grid, args...)

@inline interp_atmos_time_series(ΣJ::NamedTuple, args...) =
    interp_atmos_time_series(values(ΣJ), args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any, <:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2], args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any, <:Any, <:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2], args...) +
    interp_atmos_time_series(ΣJ[3], args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any, <:Any, <:Any, <:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2], args...) +
    interp_atmos_time_series(ΣJ[3], args...) +
    interp_atmos_time_series(ΣJ[4], args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any, <:Any, <:Any, <:Any, <:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2], args...) +
    interp_atmos_time_series(ΣJ[3], args...) +
    interp_atmos_time_series(ΣJ[4], args...) +
    interp_atmos_time_series(ΣJ[5], args...)

@inline interp_atmos_time_series(ΣJ::Tuple, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2:end], args...)
