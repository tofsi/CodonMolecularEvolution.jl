const _GENERIC_MATCH_TOKENS = Set([
    "omnibus", "multi", "omnibusmulti", "dataset", "datasets",
    "alignment", "alignments", "aln", "seq", "seqs", "sequence", "sequences",
    "tree", "trees", "newick", "nwk", "tre",
    "true", "truth", "rates", "rate", "true_rates", "truerates",
    "csv", "tsv", "txt", "fasta", "fa", "fna", "fas",
])

_slugify_string(s::AbstractString) = begin
    s2 = lowercase(strip(s))
    s2 = replace(s2, r"[^a-z0-9]+" => "_")
    s2 = replace(s2, r"_+" => "_")
    s2 = strip(s2, '_')
    isempty(s2) ? "simulation" : s2
end

function slugify_real(x::Real)
    s = replace(string(round(Float64(x), digits=6)), '-' => "m", '.' => "p")
    return replace(s, r"[^A-Za-z0-9_]" => "_")
end

read_timestamp() = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")

function append_csv_row(path::AbstractString, row)
    df = DataFrame([row])
    mkpath(dirname(path))
    if isfile(path)
        CSV.write(path, df; append=true, writeheader=false)
    else
        CSV.write(path, df)
    end
    return path
end

function read_table_auto(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext == ".tsv"
        return DataFrame(CSV.File(path; delim='\t'))
    elseif ext == ".txt"
        first_line = open(readline, path)
        if occursin('\t', first_line)
            return DataFrame(CSV.File(path; delim='\t'))
        else
            return DataFrame(CSV.File(path))
        end
    else
        return DataFrame(CSV.File(path))
    end
end

function read_fasta_simple(path::AbstractString)
    seqnames = String[]
    seqs = String[]

    current_name = Ref{Union{Nothing,String}}(nothing)
    current_seq = Ref(IOBuffer())

    seqline_rx = r"^[A-Za-z\-\?\.]+$"

    function flush_record!()
        if current_name[] !== nothing
            push!(seqnames, current_name[]::String)
            push!(seqs, String(take!(current_seq[])))
        end
    end

    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, ';') && continue

            if startswith(line, '>')
                flush_record!()
                current_name[] = strip(line[2:end])
                current_seq[] = IOBuffer()
                continue
            end

            current_name[] === nothing && continue
            seqline = replace(line, r"\s+" => "")

            if occursin(seqline_rx, seqline)
                write(current_seq[], seqline)
            else
                @warn "Ignoring non-sequence line while reading FASTA" path=path current_name=current_name[] line=line
                continue
            end
        end
    end

    flush_record!()

    isempty(seqnames) && error("No FASTA records found in $path")
    return seqnames, seqs
end

function trim_long_sequences_to_modal_length(
    seqnames::Vector{String},
    seqs::Vector{String};
    source::AbstractString="",
)
    lengths = length.(seqs)
    counts = Dict(l => count(==(l), lengths) for l in unique(lengths))
    modal_length = first(sort(collect(keys(counts)); by=l -> -counts[l]))

    fixed = copy(seqs)
    changed = false

    for i in eachindex(fixed)
        if length(fixed[i]) > modal_length
            @warn "Trimming overlong sequence to modal alignment length" source=source name=seqnames[i] old_length=length(fixed[i]) new_length=modal_length
            fixed[i] = fixed[i][1:modal_length]
            changed = true
        elseif length(fixed[i]) < modal_length
            error(
                "Sequence $(seqnames[i]) is shorter than modal alignment length in $source. " *
                "length=$(length(fixed[i])), modal_length=$modal_length"
            )
        end
    end

    return seqnames, fixed, changed
end

function validate_alignment_for_flavor(
    seqnames::Vector{String},
    seqs::Vector{String};
    source::AbstractString="",
)
    isempty(seqs) && error("No sequences read from $source")

    lengths = length.(seqs)
    unique_lengths = sort(unique(lengths))

    if length(unique_lengths) != 1
        counts = Dict(l => count(==(l), lengths) for l in unique_lengths)

        examples = String[]
        for l in unique_lengths
            idx = findfirst(==(l), lengths)
            push!(examples, "$(seqnames[idx]) length=$l")
        end

        error(
            "Alignment has unequal sequence lengths in $source.\n" *
            "Length counts: $(counts)\n" *
            "Examples: $(examples)"
        )
    end

    L = only(unique_lengths)

    if L % 3 != 0
        error(
            "Alignment length is not divisible by 3 in $source. " *
            "Length = $L. FLAVOR expects a codon alignment."
        )
    end

    bad = findall(seq -> occursin(r"[^A-Za-z\-\?\.]", seq), seqs)
    if !isempty(bad)
        i = first(bad)
        error(
            "Sequence $(seqnames[i]) in $source contains unexpected characters.\n" *
            "First 100 chars: $(seqs[i][1:min(end, 100)])"
        )
    end

    return true
end

function strip_tree_group_tags_for_flavor(treestring::AbstractString)
    return replace(String(treestring), r"\{GROUP_[0-9A-Za-z_:-]*\}" => "")
end

function _tokenize_for_matching(path::AbstractString)
    base = lowercase(splitext(basename(path))[1])
    toks = String[tok for tok in split(base, r"[^a-z0-9]+") if !isempty(tok)]
    toks = [tok for tok in toks if !(tok in _GENERIC_MATCH_TOKENS)]
    return unique(toks)
end

function _jaccard_similarity(a::AbstractVector{<:AbstractString}, b::AbstractVector{<:AbstractString})
    sa, sb = Set(String.(a)), Set(String.(b))
    isempty(sa) && isempty(sb) && return 1.0
    u = length(union(sa, sb))
    u == 0 && return 0.0
    return length(intersect(sa, sb)) / u
end

function _best_match(target_path::AbstractString, candidates::Vector{String}; min_score::Float64=0.2)
    isempty(candidates) && return nothing, 0.0
    target_tokens = _tokenize_for_matching(target_path)
    best_path = nothing
    best_score = -Inf
    for cand in candidates
        score = _jaccard_similarity(target_tokens, _tokenize_for_matching(cand))
        if score > best_score
            best_score = score
            best_path = cand
        end
    end
    if best_path === nothing || best_score < min_score
        return nothing, best_score
    end
    return best_path, best_score
end
