function threshold_summary(scores, truth::AbstractVector{Bool}, threshold::Real)
    length(scores) == length(truth) || throw(DimensionMismatch("scores and truth lengths differ"))
    called = collect(scores .>= threshold)

    tp = count(called .& truth)
    fp = count(called .& .!truth)
    fn = count(.!called .& truth)
    tn = count(.!called .& .!truth)

    tpr = (tp + fn) == 0 ? NaN : tp / (tp + fn)
    fpr = (fp + tn) == 0 ? NaN : fp / (fp + tn)
    precision = (tp + fp) == 0 ? NaN : tp / (tp + fp)

    return (
        n_called=count(called),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        tpr=tpr,
        fpr=fpr,
        precision=precision,
    )
end

function roc_curve_from_scores(scores, truth::AbstractVector{Bool})
    scores = Float64.(collect(scores))
    truth = collect(truth)
    length(scores) == length(truth) || throw(DimensionMismatch("scores and truth lengths differ"))

    thresholds = sort(unique(scores); rev=true)
    thresholds = vcat(Inf, thresholds, -Inf)

    rows = NamedTuple[]
    for th in thresholds
        called = scores .>= th
        tp = count(called .& truth)
        fp = count(called .& .!truth)
        fn = count(.!called .& truth)
        tn = count(.!called .& .!truth)
        tpr = (tp + fn) == 0 ? NaN : tp / (tp + fn)
        fpr = (fp + tn) == 0 ? NaN : fp / (fp + tn)
        push!(rows, (
            threshold=th,
            tpr=tpr,
            fpr=fpr,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
        ))
    end

    df = DataFrame(rows)
    sort!(df, [:fpr, :tpr])

    auc = 0.0
    clean = dropmissing(df[:, [:fpr, :tpr]])
    for i in 2:nrow(clean)
        dx = clean.fpr[i] - clean.fpr[i-1]
        yavg = (clean.tpr[i] + clean.tpr[i-1]) / 2
        auc += dx * yavg
    end

    return df, auc
end

function pr_curve_from_scores(scores, truth::AbstractVector{Bool})
    scores = Float64.(collect(scores))
    truth = collect(truth)
    length(scores) == length(truth) || throw(DimensionMismatch("scores and truth lengths differ"))

    thresholds = sort(unique(scores); rev=true)
    thresholds = vcat(Inf, thresholds, -Inf)

    rows = NamedTuple[]
    for th in thresholds
        called = scores .>= th
        tp = count(called .& truth)
        fp = count(called .& .!truth)
        fn = count(.!called .& truth)

        recall = (tp + fn) == 0 ? NaN : tp / (tp + fn)
        precision = (tp + fp) == 0 ? 1.0 : tp / (tp + fp)

        push!(rows, (
            threshold=th,
            precision=precision,
            recall=recall,
            tp=tp,
            fp=fp,
            fn=fn,
        ))
    end

    df = DataFrame(rows)
    sort!(df, :recall)

    auprc = 0.0
    clean = dropmissing(df[:, [:recall, :precision]])
    for i in 2:nrow(clean)
        dx = clean.recall[i] - clean.recall[i-1]
        yavg = (clean.precision[i] + clean.precision[i-1]) / 2
        auprc += dx * yavg
    end

    return df, auprc
end
