### A Pluto.jl notebook ###
# v0.18.2

using Markdown
using InteractiveUtils

# ╔═╡ b3fe06ca-a30e-11ec-1de5-4bd280e579c6
begin
	using Pkg
	Pkg.activate(".")
	
	using DataDeps

	using CSV
	using DataFrames
	using Statistics
	using Yawipa
	using Latexify
end

# ╔═╡ fb96a818-9397-479d-98d9-02db236c4ee7
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
	ENV["JULIA_NUM_THREADS"] = 16
end

# ╔═╡ bd240b14-11bf-4c69-957a-b09cefc402e1
md"""
# GCIDE Unfiltered Preprocessing
"""

# ╔═╡ a98e5705-65fa-46f7-a814-16829d85eb25
begin
	register(DataDep(
		"GCIDE",
		"""
		Dataset: GCIDE
		""",
		"https://ftp.gnu.org/gnu/gcide/gcide-0.53.tar.gz",
		"c2c6169c8247479f498ef55a2bea17d490fdb2612139bab3b8696adb1ee1a778",
		post_fetch_method=(file -> unpack(file))
	))
end

# ╔═╡ 7a699b59-3c58-4e09-bd62-85ece9fe169c
datadep"GCIDE"

# ╔═╡ 7bf8323c-cb1d-4c0b-a64a-b645359df274
function readGCIDE(s)
	res = Dict{String, Vector{SubString{String}}}()
	group_re = r"<p>(.*)</p>"
	word_re = r"<ent>(.*?)</ent>"
	def_re = r"<def>(.*?)</def>"

	for group in eachmatch(group_re, s)
		if group != nothing
			lower = lowercase(group.captures[1])
			word_match = match(word_re, lower)
			if word_match != nothing 
				# collect first word
				word = word_match.captures[1]
				def_match = eachmatch(def_re, lower)
				if def_match != nothing
					# collect all definitions
					defs = [match.captures[1] for match in def_match]
					res[word] = defs
				end
			end
		end
	end

	return res
end

# ╔═╡ f32cc1af-ded4-46c6-b986-87f6e20bcf27
function readGCIDEAll()
	res = Dict{String, Vector{SubString{String}}}()
	for ch in 'A':'Z'
		gcide = string(datadep"GCIDE", "/gcide-0.53")
		s = open(f->read(f, String), string(gcide, "/CIDE.$(ch)"))
		s = replace(s, "<br/\n" => "<br/")
		res = merge(res, readGCIDE(s))
	end
	return res
end

# ╔═╡ 9583cea8-3073-416f-aafd-7366dfc331de
md"""
# Ishiwatari Datasets Preprocessing
"""

# ╔═╡ e2d6cbdd-9fec-4ecf-881f-9a590a26feb7
register(DataDep(
	"Ishiwatari",
	"""
	Dataset: Ishiwatari
	Website: https://github.com/DefinitionModeling/ishiwatari-naacl2019
	""",
	"http://www.tkl.iis.u-tokyo.ac.jp/~ishiwatari/naacl_data.zip",
	"d8775ea04757c9281f13e9a45ef5234eec5f9eea91b5411ea0781a93e3a134b2",
	post_fetch_method=(file -> unpack(file))
))

# ╔═╡ 14ca9b7c-67ad-4d00-9d56-bd528479c0a7
datadep"Ishiwatari"

# ╔═╡ df58a7fa-3be6-4266-a185-6da2ce663626
function loadSplit(dataset, split)
	ishi = string(datadep"Ishiwatari", "/data")
	file = string(ishi, "/$(dataset)/$(split).txt")
	df = DataFrame(CSV.File(file, delim="\t", header=[
		"word", "pos", "source", "definition", "con", "desc",
	]))

	return df
end

# ╔═╡ 1fd2fe2d-f632-46d7-abe7-1f19f5ca55be
function readIshiwatari(dataset::String)
	# open file
	df = vcat(
		loadSplit(dataset, "train"),
		loadSplit(dataset, "test"),
		loadSplit(dataset, "valid"),
	)

	# remove extra word info
	re = r"%(.*)"
	df = transform(
		df,
		:word => ByRow(x -> replace(
			x,
			match(re, x).match => "",
		)) => :word,
	)

	# get polyseme column
	gdf = groupby(df, :word)
	polys = combine(gdf) do sdf
		DataFrame(polyseme = length(sdf.definition) > 1 ? true : false)
	end

	return df, polys
end

# ╔═╡ 4b2cc8b0-98d7-4b32-a521-e4fa440172f9
begin
	ishi = string(datadep"Ishiwatari", "/data")
	for file in ["train", "test", "valid"]
		out_path = string(ishi, "/slang2")
		mkpath(out_path)
		
		out_file = string(out_path, "/$(file).txt")
		if !isfile(out_path)
			in_file = string(ishi, "/slang/$(file).txt")

			# remove all quotes
			s = open(f->read(f, String), in_file)
			s = replace(s, "\"" => "")
			open(out_file, "w") do io
				write(io, s)
			end
		end
	end
end

# ╔═╡ 0cc11185-d090-452e-8f91-e6db9bcbce59
md"""
# Wiktionary Preprocessing
"""

# ╔═╡ 8b0337d5-9c16-428e-830d-ab116c5eb28c
register(DataDep("Wiktionary",
        """
        Dataset: Wiktionary
        Website: https://en.wiktionary.org/
        """,
        "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2",
        "1a86e2966f5195beeab5953e865b183a5d7f0408e3fccb4da79cf6190c7c34df";
        post_fetch_method=(file -> unpack(file, keep_originals=true))
))

# ╔═╡ af4fc4a1-b36e-458a-b704-0f3f0628d4fa
datadep"Wiktionary"

# ╔═╡ 170dfc16-66bf-4055-a226-f37cfd3ad368
function readWiktionary()
	out_path = string(datadep"Wiktionary", "/dataset")
	mkpath(out_path)
	
	out_file = string(out_path, "/yawipa.csv")
	if !isfile(out_file)
		Yawipa.parse(
			datadep"Wiktionary/enwiktionary-latest-pages-articles.xml",
			"en",
			string(out_file),
			string(out_path, "/yawipa.log"),
			".*:.*",
			["def"]
		)
	end

	df = DataFrame(CSV.File(out_file, delim="\t", header=[
		"lang", "word", "pos", "type", "definition", "info",
	]))

	return df
end

# ╔═╡ 6be8b9b0-455f-4aa5-980e-27d19d6cf769
function readWiktionaryProcessed(lang::String)
	df = readWiktionary()
	
	# filter missing data
	df = select(df, [:lang, :word, :definition])
	df = dropmissing(df)
	df = filter(:lang => ==(lang), df)
	
	# get polyseme column
	gdf = groupby(df, :word)
	polys = combine(gdf) do sdf
		DataFrame(polyseme = length(sdf.definition) > 1 ? true : false)
	end
	
	return df, polys
end

# ╔═╡ 35f5c943-5c54-44cd-b6ad-38639904f913
md"""
# Dataset Stats
"""

# ╔═╡ 55eba273-0377-4de4-8734-db66ca3e2673
function dict_to_df(dataset::Dict)
	res = DataFrame("word"=>String[], "definition"=>String[], "polyseme"=>Bool[])
	for (word, defs) in dataset
		for def in defs
			push!(res, Dict(
				"word"=>word,
				"definition"=>String(def),
				"polyseme"=>Bool(length(defs) > 1)
			))
		end
	end
	return res
end

# ╔═╡ e9df45c7-44aa-40bf-917f-3cd515687bc0
function dataset_stats(dataset)
	num_words = length(unique(dataset.word))
	num_definitions = length(dataset.definition)
	definitions_per_word = num_definitions/num_words

	return Dict{String, Any}(
		"num_words"=>num_words,
		"num_definitions"=>num_definitions,
		"definitions_per_word"=>round(definitions_per_word, digits=3),
	)
end

# ╔═╡ 23a8c3a1-ece4-4971-b818-686eb95c7608
function polyseme_stats(df::DataFrame)
	num_words = length(unique(df.word))
	
	# get number of polysemous words
	num_polysemes = length(filter(row -> row.polyseme, df).polyseme)

	return Dict{String, Float64}(
		"num_polysemes"=>num_polysemes,
		"polyseme_ratio"=>round(num_polysemes/num_words, digits=3),
	)
end

# ╔═╡ ae799722-2a64-4716-b2ab-de4fa0072261
md"""
## GCIDE Stats
"""

# ╔═╡ ff7bac23-36fa-4508-8737-817978d00cec
gcide_df = dict_to_df(readGCIDEAll());

# ╔═╡ 3acf9f9f-777a-4a30-9521-775ddb0fd567
begin
	gcide_stats = merge(dataset_stats(gcide_df), polyseme_stats(gcide_df))
	gcide_stats["dataset"] = "gcide"
	gcide_stats
end

# ╔═╡ f41e66e3-86dd-4822-bcb9-c8f133f6a355
md"""
## Oxford Stats
"""

# ╔═╡ df0f3283-cba6-4fd4-b7bf-9f8b10724fe9
oxford_df, oxford_poly = readIshiwatari("oxford");

# ╔═╡ 95e15a38-c17d-4365-8bc0-0774cdfb3127
begin
	oxford_stats = merge(dataset_stats(oxford_df), polyseme_stats(oxford_poly))
	oxford_stats["dataset"] = "oxford"
	oxford_stats
end

# ╔═╡ 502dc8ac-ce8c-4fb2-9513-4cc8c7bfd25a
md"""
## Urban Dictionary Stats
"""

# ╔═╡ 21a48539-4f67-4104-b549-2c96b5f65f00
slang_df, slang_polys = readIshiwatari("slang2");

# ╔═╡ 2dc83a51-4f54-4ff5-85bd-7df41ae2d7e1
begin
	slang_stats = merge(dataset_stats(slang_df), polyseme_stats(slang_polys))
	slang_stats["dataset"] = "urban"
	slang_stats
end

# ╔═╡ 6107b1bb-de22-4dd6-ac01-197c92309191
md"""
## Wikipedia Stats
"""

# ╔═╡ f52cf939-28a8-4470-bfcd-89371e19951a
wiki_df, wiki_polys = readIshiwatari("wiki");

# ╔═╡ 708fcf2a-a052-4ef3-92bb-60b16a4d778e
begin
	wiki_stats = merge(dataset_stats(wiki_df), polyseme_stats(wiki_polys))
	wiki_stats["dataset"] = "wikipedia"
	wiki_stats
end

# ╔═╡ 228d1f50-81ea-40ff-a67e-5833c6900750
md"""
## WordNet Stats
"""

# ╔═╡ 4200aa80-052c-4611-95de-17f13e96d2e9
wordnet_df, wordnet_polys = readIshiwatari("wordnet");

# ╔═╡ 2c3e6e56-1e7e-4eb0-9000-58fdf8e127cb
begin
	wordnet_stats = merge(dataset_stats(wordnet_df), polyseme_stats(wordnet_polys))
	wordnet_stats["dataset"] = "wordnet"
	wordnet_stats
end

# ╔═╡ d2eb52c3-9916-4f5a-bece-8dbd8c898a39
md"""
## Wiktionary English Stats
"""

# ╔═╡ 492da372-39c3-4230-a093-90aaf4f5621c
wiktionary_eng_df, wiktionary_eng_polys = readWiktionaryProcessed("eng");

# ╔═╡ 8f441a35-b518-4a5c-a49b-f19d9ed2eef2
begin
	wiktionary_eng_stats = merge(dataset_stats(wiktionary_eng_df), polyseme_stats(wiktionary_eng_polys))
	wiktionary_eng_stats["dataset"] = "wiktionary_eng"
	wiktionary_eng_stats
end

# ╔═╡ 3df8ef4c-d0af-4729-b4e2-80a3cc276afd
md"""
## Wiktionary French Stats
"""

# ╔═╡ b99d1a54-20a6-4a73-bd8a-68ebec951e33
wiktionary_fra_df, wiktionary_fra_polys = readWiktionaryProcessed("fra");

# ╔═╡ 9c2450e6-6c24-4e12-a7c9-30264f86ec9c
begin
	wiktionary_fra_stats = merge(dataset_stats(wiktionary_fra_df), polyseme_stats(wiktionary_fra_polys))
	wiktionary_fra_stats["dataset"] = "wiktionary_fra"
	wiktionary_fra_stats
end

# ╔═╡ f56acc68-b977-4cf6-a3d1-1c6c7169fb39
md"""
## Wiktionary German Stats
"""

# ╔═╡ 4d37f97b-cbe0-4119-9eb8-d461ec887b52
wiktionary_deu_df, wiktionary_deu_polys = readWiktionaryProcessed("deu");

# ╔═╡ 31d0f61d-3a8d-40c6-bcd4-ff80db9b04e9
begin
	wiktionary_deu_stats = merge(dataset_stats(wiktionary_deu_df), polyseme_stats(wiktionary_deu_polys))
	wiktionary_deu_stats["dataset"] = "wiktionary_fra"
	wiktionary_deu_stats
end

# ╔═╡ 5218330b-1a04-4ac6-a140-2656791b8984
stats_df = select(
	vcat(
		DataFrame(gcide_stats),
		DataFrame(oxford_stats),
		DataFrame(wordnet_stats),
		DataFrame(wiki_stats),
		DataFrame(slang_stats),
		DataFrame(wiktionary_eng_stats),
	), [
		:dataset, :num_words, :num_definitions, :definitions_per_word, :num_polysemes, :polyseme_ratio
	]
)

# ╔═╡ 3f609a12-99d8-4c5b-af3c-326a34e690b4
res = latexify(stats_df, env=:table)

# ╔═╡ 73796904-eaf6-47f2-b8e7-2ac537764848
vcat(
	DataFrame(wiktionary_eng_stats),
	DataFrame(wiktionary_fra_stats),
	DataFrame(wiktionary_deu_stats),
)

# ╔═╡ Cell order:
# ╠═b3fe06ca-a30e-11ec-1de5-4bd280e579c6
# ╠═fb96a818-9397-479d-98d9-02db236c4ee7
# ╟─bd240b14-11bf-4c69-957a-b09cefc402e1
# ╠═a98e5705-65fa-46f7-a814-16829d85eb25
# ╠═7a699b59-3c58-4e09-bd62-85ece9fe169c
# ╠═7bf8323c-cb1d-4c0b-a64a-b645359df274
# ╠═f32cc1af-ded4-46c6-b986-87f6e20bcf27
# ╟─9583cea8-3073-416f-aafd-7366dfc331de
# ╠═e2d6cbdd-9fec-4ecf-881f-9a590a26feb7
# ╠═14ca9b7c-67ad-4d00-9d56-bd528479c0a7
# ╠═df58a7fa-3be6-4266-a185-6da2ce663626
# ╠═1fd2fe2d-f632-46d7-abe7-1f19f5ca55be
# ╠═4b2cc8b0-98d7-4b32-a521-e4fa440172f9
# ╟─0cc11185-d090-452e-8f91-e6db9bcbce59
# ╠═8b0337d5-9c16-428e-830d-ab116c5eb28c
# ╠═af4fc4a1-b36e-458a-b704-0f3f0628d4fa
# ╠═170dfc16-66bf-4055-a226-f37cfd3ad368
# ╠═6be8b9b0-455f-4aa5-980e-27d19d6cf769
# ╟─35f5c943-5c54-44cd-b6ad-38639904f913
# ╠═55eba273-0377-4de4-8734-db66ca3e2673
# ╠═e9df45c7-44aa-40bf-917f-3cd515687bc0
# ╠═23a8c3a1-ece4-4971-b818-686eb95c7608
# ╟─ae799722-2a64-4716-b2ab-de4fa0072261
# ╠═ff7bac23-36fa-4508-8737-817978d00cec
# ╠═3acf9f9f-777a-4a30-9521-775ddb0fd567
# ╟─f41e66e3-86dd-4822-bcb9-c8f133f6a355
# ╠═df0f3283-cba6-4fd4-b7bf-9f8b10724fe9
# ╠═95e15a38-c17d-4365-8bc0-0774cdfb3127
# ╟─502dc8ac-ce8c-4fb2-9513-4cc8c7bfd25a
# ╠═21a48539-4f67-4104-b549-2c96b5f65f00
# ╠═2dc83a51-4f54-4ff5-85bd-7df41ae2d7e1
# ╟─6107b1bb-de22-4dd6-ac01-197c92309191
# ╠═f52cf939-28a8-4470-bfcd-89371e19951a
# ╠═708fcf2a-a052-4ef3-92bb-60b16a4d778e
# ╟─228d1f50-81ea-40ff-a67e-5833c6900750
# ╠═4200aa80-052c-4611-95de-17f13e96d2e9
# ╠═2c3e6e56-1e7e-4eb0-9000-58fdf8e127cb
# ╟─d2eb52c3-9916-4f5a-bece-8dbd8c898a39
# ╠═492da372-39c3-4230-a093-90aaf4f5621c
# ╠═8f441a35-b518-4a5c-a49b-f19d9ed2eef2
# ╟─3df8ef4c-d0af-4729-b4e2-80a3cc276afd
# ╠═b99d1a54-20a6-4a73-bd8a-68ebec951e33
# ╠═9c2450e6-6c24-4e12-a7c9-30264f86ec9c
# ╟─f56acc68-b977-4cf6-a3d1-1c6c7169fb39
# ╠═4d37f97b-cbe0-4119-9eb8-d461ec887b52
# ╠═31d0f61d-3a8d-40c6-bcd4-ff80db9b04e9
# ╠═5218330b-1a04-4ac6-a140-2656791b8984
# ╠═3f609a12-99d8-4c5b-af3c-326a34e690b4
# ╠═73796904-eaf6-47f2-b8e7-2ac537764848
