function _normalize_colname(x)
    s = lowercase(String(x))
    s = replace(s, r"[^a-z0-9_]+" => "_")
    s = replace(s, r"_+" => "_")
    s = strip(s, '_')
    return Symbol(s)
end

function _find_first_col(nms, regexes)
    for rx in regexes
        for nm in nms
            if occursin(rx, String(nm))
                return nm
            end
        end
    end
    return nothing
end

function _find_all_cols(nms, rx)
    return Symbol[nm for nm in nms if occursin(rx, String(nm))]
end

function truth_to_bool_vector(truth, n_sites::Integer)
    if truth isa AbstractVector{Bool}
        length(truth) == n_sites ||
            throw(DimensionMismatch("Truth Bool vector has length $(length(truth)); expected $n_sites."))
        return collect(truth)
    end

    if truth isa AbstractVector{<:Integer}
        out = falses(n_sites)
        for i in truth
            1 <= i <= n_sites || throw(BoundsError(out, i))
            out[i] = true
        end
        return out
    end

    throw(ArgumentError("truth must be either a Bool vector or a vector of 1-based positive-site indices."))
end

function _read_npz_key(path::AbstractString, key::AbstractString)
    reader = ZipFile.Reader(path)

    try
        available = String[]

        for f in reader.files
            full_name = String(f.name)
            base_name = basename(full_name)
            stem = endswith(base_name, ".npy") ? base_name[1:end-4] : base_name

            push!(available, full_name)

            if stem == key ||
               base_name == key ||
               base_name == key * ".npy" ||
               full_name == key ||
               full_name == key * ".npy" ||
               endswith(full_name, "/" * key * ".npy")

                return NPZ.npzreadarray(f)
            end
        end

        error(
            "Could not find key '$key' in $path. " *
            "Available NPZ members were: $(available)"
        )
    finally
        close(reader)
    end
end

function load_omnibus_multi_truth(
    true_rates_path::AbstractString;
    sim::Union{Nothing,Integer}=nothing,
    replicate::Union{Nothing,Integer}=nothing,
    meta_path::Union{Nothing,AbstractString}=nothing,
)
    ext = lowercase(splitext(true_rates_path)[2])

    if ext == ".npz"
        return load_omnibus_multi_truth_npz(true_rates_path; sim=sim)
    else
        raw = read_table_auto(true_rates_path)
        return load_omnibus_multi_truth_table(raw, true_rates_path)
    end
end

function load_omnibus_multi_truth_npz(
    true_rates_path::AbstractString;
    sim::Union{Nothing,Integer}=nothing,
)
    sim === nothing && error("Loading omnibus_multi_true_rates.npz requires keyword argument sim")

    sim_ids = vec(Array(_read_npz_key(true_rates_path, "sim_ids")))

    matches = findall(==(sim), sim_ids)
    isempty(matches) && error("Simulation id $sim was not found in sim_ids from $true_rates_path")
    length(matches) == 1 || error("Simulation id $sim appears multiple times in sim_ids from $true_rates_path")

    sim_ix = only(matches)

    M = try
        rates = Array(_read_npz_key(true_rates_path, "rates"))
        ndims(rates) == 3 || error("Expected rates tensor to have 3 dimensions, got size $(size(rates))")
        Float64.(rates[sim_ix, :, :])
    catch err
        rate_key = "rates_" * lpad(string(sim_ix - 1), 4, "0")
        Float64.(Array(_read_npz_key(true_rates_path, rate_key)))
    end

    ndims(M) == 2 || error("Expected truth rates matrix to be sites × parameters, got size $(size(M))")

    n_sites, n_params = size(M)
    n_params >= 2 || error("Expected alpha plus at least one non-alpha rate column, got $n_params columns.")

    raw = DataFrame(site=1:n_sites)
    raw[!, :alpha] = M[:, 1]

    for j in 2:n_params
        raw[!, Symbol("rate_$(j - 1)")] = M[:, j]
    end

    return load_omnibus_multi_truth_table(raw, true_rates_path)
end

function load_omnibus_multi_truth_table(raw::DataFrame, source_name::AbstractString)
    nrow(raw) > 0 || error("True-rates file is empty: $source_name")

    rename!(raw, Dict(nm => _normalize_colname(nm) for nm in names(raw)))

    nms = Symbol.(names(raw))

    site_col = _find_first_col(nms, [
        r"^site$",
        r"^site_index$",
        r"^codon$",
        r"^position$",
        r"^pos$",
    ])

    if site_col === nothing
        raw[!, :site] = collect(1:nrow(raw))
    elseif site_col != :site
        rename!(raw, site_col => :site)
    end

    nms = Symbol.(names(raw))

    alpha_col = _find_first_col(nms, [
        r"^alpha$",
        r"^a$",
        r"^syn_rate$",
        r"^synonymous_rate$",
    ])

    omega_cols = _find_all_cols(nms, r"(^omega([_0-9a-z]|$))|(^ω([_0-9a-z]|$))|(^w([0-9_]|$))")
    beta_cols = _find_all_cols(nms, r"(^beta([_0-9a-z]|$))|(^β([_0-9a-z]|$))|(^b([0-9_]|$))")
    rate_cols = _find_all_cols(nms, r"^rate_[0-9]+$")

    if !isempty(omega_cols)
        truth_positive = Vector{Bool}(undef, nrow(raw))
        truth_max_omega = Vector{Float64}(undef, nrow(raw))
        truth_n_positive_groups = Vector{Int}(undef, nrow(raw))

        for i in 1:nrow(raw)
            omegas = Float64[Float64(raw[i, col]) for col in omega_cols]
            truth_positive[i] = any(>(1.0), omegas)
            truth_max_omega[i] = maximum(omegas)
            truth_n_positive_groups[i] = count(>(1.0), omegas)
        end

        raw[!, :truth_signal_source] .= "omega"

    else
        alpha_col === nothing && error(
            "Could not infer alpha column from $source_name. Columns were: $(names(raw))"
        )

        nonalpha_cols = !isempty(beta_cols) ? beta_cols : rate_cols
        isempty(nonalpha_cols) && error(
            "Could not infer beta/rate columns from $source_name. Columns were: $(names(raw))"
        )

        truth_positive = Vector{Bool}(undef, nrow(raw))
        truth_max_omega = Vector{Float64}(undef, nrow(raw))
        truth_n_positive_groups = Vector{Int}(undef, nrow(raw))

        for i in 1:nrow(raw)
            α = Float64(raw[i, alpha_col])
            βs = Float64[Float64(raw[i, col]) for col in nonalpha_cols]

            truth_positive[i] = any(β -> β > α, βs)

            ωs = α == 0.0 ? fill(Inf, length(βs)) : βs ./ α
            truth_max_omega[i] = maximum(ωs)
            truth_n_positive_groups[i] = count(>(1.0), ωs)
        end

        raw[!, :truth_signal_source] .= "rate_over_alpha"
    end

    raw[!, :truth_positive] = truth_positive
    raw[!, :true_positive] = truth_positive
    raw[!, :truth_max_omega] = truth_max_omega
    raw[!, :truth_n_positive_groups] = truth_n_positive_groups

    sort!(raw, :site)

    return collect(raw.truth_positive), raw
end
