function roc_curve_from_scores(scores::AbstractVector{<:Real}, truth)
    n = length(scores)
    y = truth_to_bool_vector(truth, n)

    thresholds = vcat(Inf, sort(unique(Float64.(scores)); rev=true), -Inf)
    n_pos = count(y)
    n_neg = n - n_pos
    n_pos > 0 || throw(ArgumentError("ROC is undefined with zero positive sites"))
    n_neg > 0 || throw(ArgumentError("ROC is undefined with zero negative sites"))

    threshold_vals = Float64[]
    tpr_vals = Float64[]
    fpr_vals = Float64[]
    tp_vals = Int[]
    fp_vals = Int[]
    tn_vals = Int[]
    fn_vals = Int[]

    s = Float64.(scores)
    for thr in thresholds
        pred = s .>= thr
        tp = count(pred .& y)
        fp = count(pred .& .!y)
        fn = n_pos - tp
        tn = n_neg - fp

        push!(threshold_vals, thr)
        push!(tpr_vals, tp / n_pos)
        push!(fpr_vals, fp / n_neg)
        push!(tp_vals, tp)
        push!(fp_vals, fp)
        push!(tn_vals, tn)
        push!(fn_vals, fn)
    end

    roc_df = DataFrame(
        threshold=threshold_vals,
        tpr=tpr_vals,
        fpr=fpr_vals,
        tp=tp_vals,
        fp=fp_vals,
        tn=tn_vals,
        fn=fn_vals,
    )

    order = sortperm(roc_df.fpr)
    x = roc_df.fpr[order]
    yroc = roc_df.tpr[order]
    auc = sum((x[2:end] .- x[1:end-1]) .* (yroc[2:end] .+ yroc[1:end-1]) ./ 2)

    return roc_df, auc
end

function pr_curve_from_scores(scores::AbstractVector{<:Real}, truth)
    n = length(scores)
    y = truth_to_bool_vector(truth, n)

    thresholds = vcat(Inf, sort(unique(Float64.(scores)); rev=true), -Inf)
    n_pos = count(y)
    n_pos > 0 || throw(ArgumentError("PR is undefined with zero positive sites"))

    threshold_vals = Float64[]
    recall_vals = Float64[]
    precision_vals = Float64[]
    tp_vals = Int[]
    fp_vals = Int[]
    fn_vals = Int[]

    s = Float64.(scores)
    for thr in thresholds
        pred = s .>= thr
        tp = count(pred .& y)
        fp = count(pred .& .!y)
        fn = n_pos - tp
        precision = (tp + fp) == 0 ? 1.0 : tp / (tp + fp)
        recall = tp / n_pos
        push!(threshold_vals, thr)
        push!(recall_vals, recall)
        push!(precision_vals, precision)
        push!(tp_vals, tp)
        push!(fp_vals, fp)
        push!(fn_vals, fn)
    end

    pr_df = DataFrame(
        threshold=threshold_vals,
        recall=recall_vals,
        precision=precision_vals,
        tp=tp_vals,
        fp=fp_vals,
        fn=fn_vals,
    )

    order = sortperm(pr_df.recall)
    x = pr_df.recall[order]
    ypr = pr_df.precision[order]
    auprc = sum((x[2:end] .- x[1:end-1]) .* (ypr[2:end] .+ ypr[1:end-1]) ./ 2)
    return pr_df, auprc
end

function threshold_summary(scores::AbstractVector{<:Real}, truth, threshold::Real)
    y = truth_to_bool_vector(truth, length(scores))
    pred = Float64.(scores) .>= Float64(threshold)
    n_pos = count(y)
    n_neg = length(y) - n_pos

    tp = count(pred .& y)
    fp = count(pred .& .!y)
    fn = n_pos - tp
    tn = n_neg - fp

    tpr = n_pos > 0 ? tp / n_pos : NaN
    fpr = n_neg > 0 ? fp / n_neg : NaN
    precision = (tp + fp) > 0 ? tp / (tp + fp) : 1.0
    recall = tpr

    return (
        tp=tp, fp=fp, tn=tn, fn=fn,
        tpr=tpr, fpr=fpr,
        precision=precision, recall=recall,
        n_called=count(pred),
    )
end
