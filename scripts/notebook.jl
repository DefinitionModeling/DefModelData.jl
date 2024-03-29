### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ b3fe06ca-a30e-11ec-1de5-4bd280e579c6
begin
	using Pkg
	Pkg.activate(".")
	
	using DataDeps

	using CSV
	using DataFrames
	using Latexify
	
	using EzXML
	using JSON
	using Yawipa

	using ColorSchemes
	using Plots
	using StatsPlots
	using PyCall
	
	using Statistics
	using WordTokenizers
end

# ╔═╡ fb96a818-9397-479d-98d9-02db236c4ee7
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
	ENV["JULIA_NUM_THREADS"] = 16
end

# ╔═╡ d2c52b61-27c2-4030-b45a-3888cdadf48c
begin
	nltk = pyimport("nltk")
	wordnet = nltk.corpus.wordnet
end

# ╔═╡ bd240b14-11bf-4c69-957a-b09cefc402e1
md"""
# GCIDE Unfiltered Preprocessing
"""

# ╔═╡ a98e5705-65fa-46f7-a814-16829d85eb25
register(DataDep(
	"GCIDE",
	"""
	Dataset: GCIDE
	""",
	"https://ftp.gnu.org/gnu/gcide/gcide-0.53.tar.gz",
	"c2c6169c8247479f498ef55a2bea17d490fdb2612139bab3b8696adb1ee1a778",
	post_fetch_method=(file -> unpack(file))
))

# ╔═╡ 7a699b59-3c58-4e09-bd62-85ece9fe169c
datadep"GCIDE"

# ╔═╡ 7bf8323c-cb1d-4c0b-a64a-b645359df274
function readGCIDE(s)
	res = Dict{String, Vector{SubString{String}}}()
	group_re = r"<p>(.*)</p>"
	word_re = r"<ent>(.*?)</ent>"
	def_re = r"<def>(.*?)</def>"

	for group in eachmatch(group_re, s)
		if group !== nothing
			lower = lowercase(group.captures[1])
			word_match = match(word_re, lower)
			if word_match !== nothing 
				# collect first word
				word = word_match.captures[1]
				def_match = eachmatch(def_re, lower)
				if def_match !== nothing
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

# ╔═╡ a169194a-2da5-4cb6-ac9a-3d9dd0b74f12
md"""
# GCIDE_XML Preprocessing
"""

# ╔═╡ c30e22da-5f91-410a-8c96-9e6cc7fa00a9
register(DataDep(
	"GCIDE_XML",
	"""
	Dataset: GCIDE_XML
	""",
	"http://rali.iro.umontreal.ca/GCIDE/new-entries.zip",
	"2c0b175196cdb4235f920e75d54bfe25cc6a240c82af0b71044e44f6c3108370",
	post_fetch_method=(file -> unpack(file))
))

# ╔═╡ 7af1a46f-bfa2-44ed-aedc-ac639841882c
datadep"GCIDE_XML"

# ╔═╡ 9d74f007-7856-4a7e-a3ff-175ce69a0bef
function readGCIDEXML()
	res = Tuple{String, String}[]
	file_path = string(datadep"GCIDE_XML", "/new-entries/gcide-entries.xml")
	s = open(f->read(f, String), file_path)
	doc = parsexml(s)
	body = elements(doc.root)[2]

	for entries in eachelement(body)
		for entry in eachelement(entries)
			word = ""
			for field in eachelement(entry)
				if field.name == "hw"
					word = field.content
				end

				if field.name == "sn"
					for sn_field in eachelement(field)
						if sn_field.name == "def"
							push!(res, (word, sn_field.content))
						end
					end
				end
			end
		end
	end
	
	return res
end

# ╔═╡ bf55adef-7f04-473a-b55b-eb7a2edc80f7
gcide_xml = readGCIDEXML()

# ╔═╡ ddbe8891-5c71-43a8-9009-1a15a8f224b0
md"""
# Noraset Dataset Preprocessing
"""

# ╔═╡ 76048fa5-25f5-44d5-aa82-a9c5c6fa5647
register(DataDep(
	"Noraset",
	"""
	Dataset: Noraset
	Website: https://github.com/DefinitionModeling/torch-defseq
	""",
	"https://github.com/DefinitionModeling/torch-defseq/archive/master.tar.gz",
	"94d4a4257038ef0c9865d7eb6564134629d115efaa80d01a19004d17f99eb038",
	post_fetch_method=(file -> unpack(file))
))

# ╔═╡ 918cc3eb-05f9-421e-8770-eb88188128dc
datadep"Noraset"

# ╔═╡ a48aad67-cfa9-4d10-a311-add65d2f0781
function loadNoraset(dataset::String)
	path = string(datadep"Noraset","/torch-defseq-master/data/commondefs")
	file = string(path, "/$(dataset).txt")
	df = DataFrame(CSV.File(file, delim="\t", header=[
		"word", "pos", "source", "definition"
	]))
	return df
end

# ╔═╡ bb334f53-78f4-4d58-9605-90bf9dfe85cd
function readNoraset()
	df = vcat(
		loadNoraset("train"),
		loadNoraset("test"),
		loadNoraset("valid"),
	)

	# get polyseme column
	gdf = groupby(df, :word)
	polys = combine(gdf) do sdf
		DataFrame(polyseme = length(sdf.definition) > 1 ? true : false)
	end

	return df, polys
end

# ╔═╡ 961ca493-f948-40b6-a558-ab6290c07bdd
#vocab
begin
	nora_path = string(datadep"Noraset","/torch-defseq-master/data/commondefs")
	nora_file = string(nora_path,"/vocab.txt")
	vocab = readlines(nora_file)[5:end]
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
function loadSplit(
	dataset::String,
	split::String,
	headers::Vector{String},
)
	file = string(dataset, "/$(split).txt")
	df = DataFrame(CSV.File(file, delim="\t", header=headers))
	return df
end

# ╔═╡ 1fd2fe2d-f632-46d7-abe7-1f19f5ca55be
function readIshiwatari(dataset::String)
	ishi = string(datadep"Ishiwatari", "/data/$(dataset)")
	headers = [
		"word", "pos", "source", "definition", "con", "desc",
	]
	# open file
	df = vcat(
		loadSplit(ishi, "train", headers),
		loadSplit(ishi, "test", headers),
		loadSplit(ishi, "valid", headers),
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
	dropmissing!(df)

	# get polyseme column
	gdf = groupby(df, :word)
	polys = combine(gdf) do sdf
		DataFrame(polyseme = length(sdf.definition) > 1 ? true : false)
	end
	
	return df, polys
end

# ╔═╡ 723796c0-cdd1-498e-8927-4543cd26d1df
md"""
# Kabiri Datasets Preprocessing
"""

# ╔═╡ d94161ca-110b-4eaa-8362-67d0a54068af
register(DataDep(
	"Kabiri",
	"""
	Dataset: Kabiri
	Website: https://github.com/DefinitionModeling/Multi-sense-Multi-lingual-Definition-Modeling
	""",
	"https://github.com/DefinitionModeling/Multi-sense-Multi-lingual-Definition-Modeling/archive/master.tar.gz",
	"b95d1cd007eba7b4e734dc93692cab517a13ef86402dcb9bde9aefb76d041a98",
	post_fetch_method=(file -> unpack(file))
))

# ╔═╡ 6a59bd35-d939-4e68-bdab-4fe72cdba7eb
datadep"Kabiri"

# ╔═╡ e4ab6b4f-ca84-4d31-944f-e98f55aa1917
function readKabiri(dataset::String)
	kabiri = string(datadep"Kabiri", "/Multi-sense-Multi-lingual-Definition-Modeling-master/DataSets/English/$(dataset)")
	headers = [
		"word", "pos", "source", "definition",
	]
	
	# open file
	df = vcat(
		loadSplit(kabiri, "train", headers),
		loadSplit(kabiri, "test", headers),
		loadSplit(kabiri, "valid", headers),
	)

	# get polyseme column
	gdf = groupby(df, :word)
	polys = combine(gdf) do sdf
		DataFrame(polyseme = length(sdf.definition) > 1 ? true : false)
	end

	return df, polys
end

# ╔═╡ 86084ad2-2b4a-4312-bd50-21e1e70c232f
md"""
# Bevilacqua Datasets Preprocessing
"""

# ╔═╡ 7f2b3999-3ed4-4dc2-bf2a-83cafc2539c5
register(DataDep(
	"Bevilacqua",
	"""
	Dataset: Bevilacqua
	Website: https://sapienzanlp.github.io/generationary-web/
	""",
	"https://sapienzanlp.github.io/generationary-web/res/HEI++1.0_SamplEval.zip",
	"e5b1335d92bc9c4a3659d962147f72c547125dac7333b9266ffa07d9da24f450",
	post_fetch_method=(file -> unpack(file))
))

# ╔═╡ 80d6fb56-de40-4241-ba3a-a60945a80fe6
datadep"Bevilacqua"

# ╔═╡ c48009a0-0a34-4677-bc7a-cb13f1e36201
function readBevilacqua()
	bevil = string(datadep"Bevilacqua")
	headers = [
		"word", "definition",
	]
	
	# open file
	df = loadSplit(bevil, "HEI++1.0", headers)
	delete!(df, 1)
	
	# get polyseme column
	gdf = groupby(df, :word)
	polys = combine(gdf) do sdf
		DataFrame(polyseme = length(sdf.definition) > 1 ? true : false)
	end
	
	return df, polys
end

# ╔═╡ 45953cff-df0e-444e-8ac1-96caf8137c99
md"""
# Reid Dataset Preprocessing
"""

# ╔═╡ a520f122-57f7-4843-b01d-b80ded800f7d
register(DataDep(
	"Reid",
	"""
	Dataset: Reid
	Website: https://machelreid.github.io
	""",
	"https://machelreid.github.io/resources/Reid2020VCDM.zip",
	"40d9abc7206a014a79623d145c7f670923c5ed723065143abb43fec9ef695647",
	post_fetch_method=(file -> unpack(file))
))

# ╔═╡ 633a9633-86ee-4432-ad19-491a4d7c9ff9
datadep"Reid"

# ╔═╡ 06883aaa-98fa-4725-a0fa-f2ba56894955
begin
	dataset = "oxford"
	split = "test"
	reid_file = string(datadep"Reid", "/Reid20VCDM/$(dataset)/_$(split).json")
	s = open(f->read(f, String), reid_file)
end

# ╔═╡ ac998f10-18bc-4a6a-8b68-9e58636d3ced
function loadSplitJson(
	dataset::String,
	split::String,
)::DataFrame
	file = string(dataset, "/_$(split).json")
	res = DataFrame(word = String[], definition = String[])
	
	for line in readlines(file)
		entry = JSON.parse(line)
		word = entry["char"]
		def = entry["definition"]
		push!(res, (word, def))
	end
	
	return res
end

# ╔═╡ 28ef61fd-c23a-4e6c-b9ec-6790ebefe4fc
function readReid(dataset::String)
	reid = string(datadep"Reid", "/Reid20VCDM/$(dataset)")

	df = vcat(
		loadSplitJson(reid, "train"),
		loadSplitJson(reid, "test"),
		loadSplitJson(reid, "valid"),
	)

	# get polyseme column
	gdf = groupby(df, :word)
	polys = combine(gdf) do sdf
		DataFrame(polyseme = length(sdf.definition) > 1 ? true : false)
	end
	
	return df, polys
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
function dict_to_df(dataset::Dict)::DataFrame
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
function dataset_stats(dataset)::Dict{String, Any}
	num_words = length(unique(dataset.word))
	num_definitions = length(dataset.definition)
	definitions_per_word = num_definitions/num_words

	# get average definition length by number of tokens
	dataset = transform(
		dataset,
		:definition => ByRow(x -> length(tokenize(x))) => :definition_length,
	)
	average_definition_length = mean(dataset.definition_length)
	std_definition_length = std(dataset.definition_length)

	return Dict{String, Any}(
		"num_words"=>num_words,
		"num_definitions"=>num_definitions,
		"definitions_per_word"=>round(definitions_per_word, digits=2),
		"average_definition_length"=>round(average_definition_length, digits=2),
		"stddev_definition_length"=>round(std_definition_length, digits=2),
	)
end

# ╔═╡ 23a8c3a1-ece4-4971-b818-686eb95c7608
function polyseme_stats(df::DataFrame)::Dict{String, Any}
	num_words = length(unique(df.word))
	
	# get number of polysemous words
	num_polysemes = length(filter(row -> row.polyseme, df).polyseme)

	return Dict{String, Any}(
		"num_polysemes"=>Int.(num_polysemes),
		"polyseme_ratio"=>round(100*num_polysemes/num_words, digits=2),
	)
end

# ╔═╡ 1d0fb198-1d89-443e-881d-4ecd64673fb8
md"""
# Benchmark Dataset Stats
"""

# ╔═╡ bb18cce0-b79f-4427-a152-ba7561c44966
function one_word(str::String)::Bool
	return !(length(tokenize(str)) > 1)
end

# ╔═╡ ebb17d85-98c1-4470-bcd0-69e54278ca9f
function nonsymbol_token(def::String)::String
	# return first nonsymbol token from a sentence
	for token in tokenize(def)
		if occursin(r"^[a-zA-Z]+$", token)
			return token
		end	
	end
	return ""
end

# ╔═╡ aab6c361-1b94-465e-99b0-fa60d5816e05
function def_analysis(df::DataFrame)::DataFrame
	# add analysis columns for definition analysis
	df[!, :first_token] = nonsymbol_token.(df[!, :definition])
	df[!, :one_word] = one_word.(df[!, :definition])
	return df
end

# ╔═╡ d2955e46-e745-45ba-a744-531302ca8394
md"""
## Noraset Stats
"""

# ╔═╡ f39371ab-77ed-4633-9f37-43c50e01e7ee
begin
	nora_df, nora_poly = readNoraset();
	nora_df = def_analysis(nora_df);
end;

# ╔═╡ 60d8d452-b6ec-49d2-a138-00003dd0fc4b
begin
	nora_stats = merge(dataset_stats(nora_df), polyseme_stats(nora_poly))
	nora_stats["dataset"] = "wordnet-nor"
	nora_stats
end

# ╔═╡ 72cba374-35db-48b7-9077-3bcc27e00ab0
md"""
## Ishiwatari Dataset Stats
"""

# ╔═╡ f41e66e3-86dd-4822-bcb9-c8f133f6a355
md"""
### Oxford Stats
"""

# ╔═╡ df0f3283-cba6-4fd4-b7bf-9f8b10724fe9
begin
	oxfordi_df, oxfordi_poly = readIshiwatari("oxford");
	oxfordi_df = def_analysis(oxfordi_df);
end;

# ╔═╡ 95e15a38-c17d-4365-8bc0-0774cdfb3127
begin
	oxfordi_stats = merge(dataset_stats(oxfordi_df), polyseme_stats(oxfordi_poly))
	oxfordi_stats["dataset"] = "oxford-gad"
	oxfordi_stats
end

# ╔═╡ 502dc8ac-ce8c-4fb2-9513-4cc8c7bfd25a
md"""
### Urban Dictionary Stats
"""

# ╔═╡ 21a48539-4f67-4104-b549-2c96b5f65f00
begin
	slangi_df, slangi_polys = readIshiwatari("slang2");
	slangi_df = def_analysis(slangi_df);
end;

# ╔═╡ 2dc83a51-4f54-4ff5-85bd-7df41ae2d7e1
begin
	slangi_stats = merge(dataset_stats(slangi_df), polyseme_stats(slangi_polys))
	slangi_stats["dataset"] = "urban-ni"
	slangi_stats
end

# ╔═╡ 6107b1bb-de22-4dd6-ac01-197c92309191
md"""
### Wikipedia Stats
"""

# ╔═╡ f52cf939-28a8-4470-bfcd-89371e19951a
begin
	wikii_df, wikii_polys = readIshiwatari("wiki");
	wikii_df = def_analysis(wikii_df);
end;

# ╔═╡ 708fcf2a-a052-4ef3-92bb-60b16a4d778e
begin
	wikii_stats = merge(dataset_stats(wikii_df), polyseme_stats(wikii_polys))
	wikii_stats["dataset"] = "wikipedia-ish"
	wikii_stats
end

# ╔═╡ 228d1f50-81ea-40ff-a67e-5833c6900750
md"""
### WordNet Stats
"""

# ╔═╡ 4200aa80-052c-4611-95de-17f13e96d2e9
begin
	wordneti_df, wordneti_polys = readIshiwatari("wordnet");
	wordneti_df = def_analysis(wordneti_df);
end;

# ╔═╡ 2c3e6e56-1e7e-4eb0-9000-58fdf8e127cb
begin
	wordneti_stats = merge(dataset_stats(wordneti_df), polyseme_stats(wordneti_polys))
	wordneti_stats["dataset"] = "wordnet-ish"
	wordneti_stats
end

# ╔═╡ 3f1c8874-260e-47f6-99bd-305c17e55487
md"""
## Kabiri Dataset Stats
"""

# ╔═╡ 8eea0d2d-870e-44a5-a2f0-1a3ce5bffd39
md"""
### WordNet Stats
"""

# ╔═╡ 8a8623b0-70d1-4f69-9804-d2963d18b8ef
begin
	wordnetk_df, wordnetk_polys = readKabiri("WordNet/2-Reconstructed-Downsampled");
	wordnetk_df = def_analysis(wordnetk_df);
end;

# ╔═╡ 2a0a3d9c-bbcd-4fd7-b6c6-fe3302a035ef
begin
	wordnetk_stats = merge(dataset_stats(wordnetk_df), polyseme_stats(wordnetk_polys))
	wordnetk_stats["dataset"] = "wordnet-kab"
	wordnetk_stats
end

# ╔═╡ 275b9b34-146f-4f9a-be5e-14468841121e
md"""
### Wiktionary Stats
"""

# ╔═╡ 741971a7-726c-48a3-81e1-a9653467c3b8
begin
	wiktionaryk_df, wiktionaryk_polys = readKabiri("Wiktionary/2-reconstructed-downsampled");
	wiktionaryk_df = def_analysis(wiktionaryk_df);
end;

# ╔═╡ 57f7faf7-347e-430e-9f4f-6f2c993e826e
begin
	wiktionaryk_stats = merge(dataset_stats(wiktionaryk_df), polyseme_stats(wiktionaryk_polys))
	wiktionaryk_stats["dataset"] = "wiktionary-kab"
	wiktionaryk_stats
end

# ╔═╡ 21bd6d8f-af6f-40a3-a162-61a0c44f8065
md"""
### Omega Stats
"""

# ╔═╡ a600c797-7d09-47c1-8ca3-c872ec19fd78
begin
	omegak_df, omegak_polys = readKabiri("Omega/2-reconstructed-downsampled");
	omegak_df = def_analysis(omegak_df);
end;

# ╔═╡ 99ae3b96-03e3-45fb-86f9-ce61fc24bd82
begin
	omegak_stats = merge(dataset_stats(omegak_df), polyseme_stats(omegak_polys))
	omegak_stats["dataset"] = "omega-kab"
	omegak_stats
end

# ╔═╡ fee640fe-f35f-4a6f-9e7b-f1315cc93af4
md"""
## Bevilacqua Dataset Stats
"""

# ╔═╡ a09f51e6-903e-4a12-9ad0-4bd05182d8f8
md"""
### Hei++ Dataset Stats
"""

# ╔═╡ d3be4181-674e-46d9-b39b-9782e42790b0
begin
	hei_df, hei_polys = readBevilacqua()
	hei_df = def_analysis(hei_df)
end;

# ╔═╡ 715f81ab-c5fc-4a99-aaab-3e034c0dea43
begin
	hei_stats = merge(dataset_stats(hei_df), polyseme_stats(hei_polys))
	hei_stats["dataset"] = "hei++-bev"
	hei_stats
end

# ╔═╡ f1ead630-7679-4144-9764-973428111edc
md"""
## Reid Dataset Stats
"""

# ╔═╡ a61e7f6c-c553-4494-9c60-10b6543bf586
# md"""
# ### Oxford Dataset Stats
# """

# ╔═╡ d02bf5b4-9017-4177-8937-0a2eef03bff5
# oxfordr_df, oxfordr_polys = readReid("oxford");

# ╔═╡ 6ae940c0-11a1-4d30-a2ea-9779587005ce
# begin
# 	oxfordr_stats = merge(dataset_stats(oxfordr_df), polyseme_stats(oxfordr_polys))
# 	oxfordr_stats["dataset"] = "oxford-reid"
# 	oxfordr_stats
# end

# ╔═╡ 3fe96326-37eb-4996-b8ea-9557af0b5516
md"""
### Wikipedia Dataset Stats
"""

# ╔═╡ a83e980e-f826-4c20-826b-017f067512a6
# wikir_df, wikir_polys = readReid("wiki");

# ╔═╡ 9072c532-9da3-4c11-9cbe-532de868b2e8
# begin
# 	wikir_stats = merge(dataset_stats(wikir_df), polyseme_stats(wikir_polys))
# 	wikir_stats["dataset"] = "wikipedia-reid"
# 	wikir_stats
# end

# ╔═╡ 39de496c-9d01-4a8e-bd98-f74ea45f5ff7
md"""
### Urban Dictionary Dataset Stats
"""

# ╔═╡ 2fefd8da-9906-4e8e-826c-761a50b0b633
# slangr_df, slangr_polys = readReid("slang");

# ╔═╡ 56ee17bb-f197-41a6-98fe-9fd0bef87278
# begin
# 	slangr_stats = merge(dataset_stats(slangr_df), polyseme_stats(slangr_polys))
# 	slangr_stats["dataset"] = "urban-reid"
# 	slangr_stats
# end

# ╔═╡ f7eb3e18-5217-4b6e-b178-575f04e8b650
md"""
# Raw Dataset Stats
"""

# ╔═╡ ae799722-2a64-4716-b2ab-de4fa0072261
md"""
## GCIDE Stats
"""

# ╔═╡ ff7bac23-36fa-4508-8737-817978d00cec
# gcide_df = dict_to_df(readGCIDEAll());

# ╔═╡ 3acf9f9f-777a-4a30-9521-775ddb0fd567
# begin
# 	gcide_stats = merge(dataset_stats(gcide_df), polyseme_stats(gcide_df))
# 	gcide_stats["dataset"] = "gcide-raw"
# 	gcide_stats
# end

# ╔═╡ d2eb52c3-9916-4f5a-bece-8dbd8c898a39
md"""
## Wiktionary English Stats
"""

# ╔═╡ 492da372-39c3-4230-a093-90aaf4f5621c
# wiktionary_eng_df, wiktionary_eng_polys = readWiktionaryProcessed("eng");

# ╔═╡ 8f441a35-b518-4a5c-a49b-f19d9ed2eef2
# begin
# 	wiktionary_eng_stats = merge(dataset_stats(wiktionary_eng_df), polyseme_stats(wiktionary_eng_polys))
# 	wiktionary_eng_stats["dataset"] = "wiktionary-eng-raw"
# 	wiktionary_eng_stats
# end

# ╔═╡ 3df8ef4c-d0af-4729-b4e2-80a3cc276afd
md"""
## Wiktionary French Stats
"""

# ╔═╡ b99d1a54-20a6-4a73-bd8a-68ebec951e33
# wiktionary_fra_df, wiktionary_fra_polys = readWiktionaryProcessed("fra");

# ╔═╡ 9c2450e6-6c24-4e12-a7c9-30264f86ec9c
# begin
# 	wiktionary_fra_stats = merge(dataset_stats(wiktionary_fra_df), polyseme_stats(wiktionary_fra_polys))
# 	wiktionary_fra_stats["dataset"] = "wiktionary-fra-raw"
# 	wiktionary_fra_stats
# end

# ╔═╡ f56acc68-b977-4cf6-a3d1-1c6c7169fb39
md"""
## Wiktionary German Stats
"""

# ╔═╡ 4d37f97b-cbe0-4119-9eb8-d461ec887b52
# wiktionary_deu_df, wiktionary_deu_polys = readWiktionaryProcessed("deu");

# ╔═╡ 31d0f61d-3a8d-40c6-bcd4-ff80db9b04e9
# begin
# 	wiktionary_deu_stats = merge(dataset_stats(wiktionary_deu_df), polyseme_stats(wiktionary_deu_polys))
# 	wiktionary_deu_stats["dataset"] = "wiktionary-ger-raw"
# 	wiktionary_deu_stats
# end

# ╔═╡ 5218330b-1a04-4ac6-a140-2656791b8984
begin
	stats_df = select(
		vcat(
			DataFrame(nora_stats),
			DataFrame(oxfordi_stats),
			DataFrame(wordneti_stats),
			DataFrame(wikii_stats),
			DataFrame(slangi_stats),
			
			DataFrame(wordnetk_stats),
			DataFrame(wiktionaryk_stats),
			DataFrame(omegak_stats),
			
			DataFrame(hei_stats),
			
			# DataFrame(oxfordr_stats),
			# DataFrame(wikir_stats),
			# DataFrame(slangr_stats),
		), [
			:dataset, :num_words, :num_definitions, :definitions_per_word, :num_polysemes, :polyseme_ratio,
			:average_definition_length, :stddev_definition_length,
		]
	)
	stats_df = sort(stats_df, :num_words)
end

# ╔═╡ 2ac8b154-2f37-472b-bf79-a73d6f0c37d9
md"""
# Analysis
"""

# ╔═╡ dbc8c73e-4d1a-4f64-8d37-5662529aa393
md"""
## Polyseme Analysis
"""

# ╔═╡ d9e23ae2-bf40-46e4-85a4-225912d4cbe4
begin
	poly_stats_df = sort(stats_df, :polyseme_ratio, rev=true)
	polysemes_plot = bar(
		poly_stats_df.dataset,
		poly_stats_df.polyseme_ratio,
		ylabel="Polysemous phrases (%)",
		xrotation=20,
		legend=false,
		palette = palette(:Dark2_8),
	)
end

# ╔═╡ 5a8d300e-01fd-4ba7-b6a0-f5186c1291a2
all_df_names = [
	nora_stats["dataset"],
	
	oxfordi_stats["dataset"],
	wordneti_stats["dataset"],
	wikii_stats["dataset"],
	slangi_stats["dataset"],

	wordnetk_stats["dataset"],
	wiktionaryk_stats["dataset"],
	omegak_stats["dataset"],
	
	hei_stats["dataset"],
]

# ╔═╡ bf0afa7a-be35-4c1b-82e3-2817c8d99d91
md"""
# Word Count Analysis
"""

# ╔═╡ 76a4cc6e-969e-4bc4-a193-39b5c5aafba0
begin
	# count words by number of appearances in each set (multiple appearances in a single set ignored)
	words = [
		unique(nora_df.word),

		unique(oxfordi_df.word),
		unique(wordneti_df.word),
		unique(wikii_df.word),
		unique(slangi_df.word),

		unique(wordnetk_df.word),
		unique(wiktionaryk_df.word),
		unique(omegak_df.word),
		
		unique(hei_df.word),
	]
	words = vcat(words...)
	words_df = DataFrame(word=String[])
	for word in words
		push!(words_df, (word,))
	end
	
	gdf = groupby(words_df, :word)
	word_counts = combine(gdf) do sdf
		DataFrame(counts = length(sdf.word))
	end

	# word_counts = filter(row -> row.counts >= 7, word_counts)
	word_counts = sort(word_counts, [:word])
	word_counts = sort(word_counts, [:counts], rev = true)
end

# ╔═╡ 1b645f58-6f5c-43f4-a759-4b9aff02c921
begin
	# get number of each count
	num_counts = combine(groupby(word_counts, :counts)) do sdf
		DataFrame(num_count = length(sdf.counts))
	end
	
	num_counts = sort(num_counts, :num_count)
end

# ╔═╡ 3c38e58f-c9b8-41ed-bbdb-73738295aeaf
unique_counts_plot = bar(
	num_counts.counts,
	num_counts.num_count,
	ylabel="Number",
	legend=false,
	palette = palette(:Dark2_8),
	yaxis=:log,
	yticks = [1, 10, 100, 1000, 10000, 100000]
)

# ╔═╡ 5329894b-0c64-4d6d-b8cd-5214e938205a
md"""
## Most Common Word Analysis
"""

# ╔═╡ ca87b2dc-621a-4431-bff1-834d16314004
begin
	# count words by number of appearances in each set (multiple appearances in a single set ignored)
	word = "movement"
	filter_word(df::DataFrame) = unique(select(
		filter(:word => ==(word), df),
		[:word, :definition],
	)).definition
	cwords = [
		nora_df,

		oxfordi_df,
		wordneti_df,
		wikii_df,
		slangi_df,

		wordnetk_df,
		wiktionaryk_df,
		omegak_df,
		
		hei_df,
	]
	cwords = filter_word.(cwords)
end

# ╔═╡ 8c15216b-17b4-43b0-821a-57cf602c3c4e
begin
	cword_df = DataFrame(dataset=String[], definition=String[])
	push!(cword_df, (all_df_names[1], cwords[1][1]))
	push!(cword_df, (all_df_names[2], cwords[2][1]))
	push!(cword_df, (all_df_names[3], cwords[3][2]))
	push!(cword_df, (all_df_names[4], cwords[4][2]))
	push!(cword_df, (all_df_names[5], cwords[5][1]))
	push!(cword_df, (all_df_names[6], cwords[6][1]))
	push!(cword_df, (all_df_names[7], cwords[7][1]))
	push!(cword_df, (all_df_names[8], cwords[8][1]))
end

# ╔═╡ 7f3ef8e7-dd72-4645-8048-968f596ed73e
cword_df.definition

# ╔═╡ eb751ce5-df60-46b5-9545-97cd13ad9a95
md"""
## Overlap Analysis
"""

# ╔═╡ 91dff8f9-e826-465b-9cd4-2666b2c972cf
begin
	function get_overlap(df1::DataFrame, df2::DataFrame)::Vector{String}
		overlap = intersect(df1.word, df2.word)
		return String.(overlap)
	end
	
	function get_overlap(df1::DataFrame, words::Vector{<:AbstractString})
		return intersect(df1.word, words)
	end
	
	function get_overlap_percent(df1::DataFrame, df2::DataFrame)::Float64
		# get overlapping word as percentage of total words in df1
		return round(
			100*length(get_overlap(df1, df2))/length(df1.word),
			digits=2,
		)
	end
end

# ╔═╡ 02a18190-3955-463a-8bad-fd7f92257075
begin
	dfs = [
		nora_df,
		
		oxfordi_df,
		wordneti_df,
		wikii_df,
		slangi_df,

		wordnetk_df,
		wiktionaryk_df,
		omegak_df,
		
		# hei_df,
	]

	df_names = [
		nora_stats["dataset"],
		
		oxfordi_stats["dataset"],
		wordneti_stats["dataset"],
		wikii_stats["dataset"],
		slangi_stats["dataset"],

		wordnetk_stats["dataset"],
		wiktionaryk_stats["dataset"],
		omegak_stats["dataset"],
		
		# hei_stats["dataset"],
	]

	df = DataFrame(df1 = String[], df2 = String[], overlap_pct = Float64[])
	for i in 1:length(dfs)
		for j in 1:length(dfs)
			if i != j
				push!(df, (
					df_names[i],
					df_names[j],
					get_overlap_percent(dfs[i], dfs[j]),
				))
			end
		end
	end
	
	df = sort(df, :overlap_pct, rev = true)
end

# ╔═╡ 8b098243-2467-40fb-9836-3750008312d1
function smallbar(df::DataFrame)
	groupedbar(
		df.df1,
		df.overlap_pct,
		group=df.df2,
		ylabel="Overlap (%)",
		ylims=(0, 55),
		xrotation=20,
		legend=:topleft,
		palette = palette(:Dark2_8),
	)
end

# ╔═╡ 4b351ddc-0927-4128-aaf4-43794a845762
grp1 = [
	omegak_stats["dataset"],
	wordneti_stats["dataset"],
	wordnetk_stats["dataset"],
	oxfordi_stats["dataset"],
]

# ╔═╡ 9dd70ccb-d434-440d-8da5-f1caf3c0dd5d
begin
	overlap_all_plot = 	groupedbar(
		df.df1,
		df.overlap_pct,
		group=df.df2,
		ylabel="Overlap (%)",
		ylims=(0, 55),
		xrotation=20,
		legend=:topright,
		size=(800, 300),
		bottom_margin = 4Plots.mm,
		left_margin = 3Plots.mm,
		palette = palette(:Dark2_8),
	)
end

# ╔═╡ 6654623f-0ae9-4c5f-8820-5957b87f0225
begin
	overlap_group1_plot = smallbar(filter(row -> row.df1 ∈ grp1, df))
end

# ╔═╡ 310b2b16-7c2d-43c1-a47a-17b0f1e58bad
begin
	overlap_group2_plot = smallbar(filter(row -> row.df1 ∉ grp1, df))
end

# ╔═╡ 3eba3f16-d7d9-4af2-907d-38c445d7a978
# average overlap
begin
	groupby(df, :df1)
	avg_overlaps = combine(groupby(df, :df1)) do sdf
		DataFrame(avg_overlap = round(
			sum(sdf.overlap_pct) / length(sdf.overlap_pct),
			digits=2,
		))
	end
	push!(avg_overlaps, (hei_stats["dataset"], 0.0))
	sorted_overlaps = sort(avg_overlaps, :avg_overlap, rev=true)
end

# ╔═╡ d5f4a207-a5dc-4636-96fd-d6eb37a16c8d
avg_overlap_plot = bar(
	sorted_overlaps.df1,
	sorted_overlaps.avg_overlap,
	ylabel="Average overlap (%)",
	xrotation=20,
	legend=false,
	palette = palette(:Dark2_8),
)

# ╔═╡ ac35abd6-2e14-40bc-9f08-0b82d7b33ee1
md"""
## Definition Analysis
"""

# ╔═╡ b4cb80f5-ff4b-4b87-8442-bb57e0ba0884
function contains_of(word::String)::Bool
	return word ∈ ["of", "about", "like"]
end

# ╔═╡ 75325322-d3b1-4fac-85ca-dc9c5b3f4a7a
function get_ofs(df::DataFrame)::Int
	return sum(contains_of.(df[!, :first_token]))
end

# ╔═╡ 9f5ff836-df46-4214-8212-22edd21054d8
begin
	defdfs = [
		nora_df,
		
		oxfordi_df,
		wordneti_df,
		wikii_df,
		slangi_df,

		wordnetk_df,
		wiktionaryk_df,
		omegak_df,
		
		hei_df,
	]

	defdf_names = [
		nora_stats["dataset"],
		
		oxfordi_stats["dataset"],
		wordneti_stats["dataset"],
		wikii_stats["dataset"],
		slangi_stats["dataset"],

		wordnetk_stats["dataset"],
		wiktionaryk_stats["dataset"],
		omegak_stats["dataset"],
		
		hei_stats["dataset"],
	]

	defdf = DataFrame(
		ds=String[],
		num_ofs=Int[],
		pct_ofs=Float64[],
		num_singles=Int[],
		pct_singles=Float64[],
	)
	for (data, name) in zip(defdfs, defdf_names)
		num_ofs = get_ofs(data)
		pct_ofs = 100*num_ofs / length(data.definition)

		num_singles = sum(data.one_word)
		pct_singles = 100*num_singles / length(data.definition)
		push!(defdf, (
			name,
			num_ofs,
			round(pct_ofs, digits=2),
			num_singles,
			round(pct_singles, digits=2),
		))
	end

	defdf
end

# ╔═╡ ffa82e4d-71f0-4066-8f64-1178d35ca285
sort(select(
	defdf,
	[
		:ds, :num_ofs, :pct_ofs,
	]
), :pct_ofs, rev = true)

# ╔═╡ 0effd27e-d000-4aad-b7db-1dbebdc2dceb
singles_df = sort(select(
	defdf,
	[
		:ds, :num_singles, :pct_singles,
	]
), :pct_singles, rev = true)

# ╔═╡ f685fb61-9b86-4a2b-93bc-41d5f8bba763
single_word_plot = bar(
	singles_df.ds,
	singles_df.pct_singles,
	ylabel="Single-word definition (%)",
	xrotation=20,
	legend=false,
	palette = palette(:Dark2_8),
)

# ╔═╡ 087f2cde-9fe6-4255-9ffd-b4aafeeee2f4
md"""
## Synonyms Analysis
"""

# ╔═╡ 834789f8-7e31-4bfb-a738-9fd9d9b2d5b3
function filter_single(df::DataFrame)::DataFrame
	return num_polysemes = filter(row -> row.one_word, df)
end

# ╔═╡ b3f9193f-6e73-4824-9269-1fd65c115750
function get_synonyms(word)::Vector{String}
	synsets = wordnet.synsets(word)
	res = []
	
	for synset in synsets
		push!(res, synset.lemma_names())
	end
	
	return unique(vcat(res...))
end

# ╔═╡ 54e23520-4348-42ab-8545-59475005dbe8
function is_synonym(w1, w2)
	return w2 ∈ get_synonyms(w1)
end

# ╔═╡ ae280702-b9aa-41c8-a597-125dacad1bab
function get_is_synonyms(df::DataFrame)::Int
	return sum(is_synonym.(df[!, :word], df[!, :first_token]))
end

# ╔═╡ 5df06c9e-1fdd-455f-9330-a44761824eb1
nora_df

# ╔═╡ dffd4b78-b502-4a69-b0c3-6961cca9632e
#check single word definitinons are a synonym of the word they define
begin
	syndfs = [
		nora_df,
		
		oxfordi_df,
		wordneti_df,
		wikii_df,
		slangi_df,

		wordnetk_df,
		wiktionaryk_df,
		omegak_df,
		
		hei_df,
	]

	syndf_names = [
		nora_stats["dataset"],
		
		oxfordi_stats["dataset"],
		wordneti_stats["dataset"],
		wikii_stats["dataset"],
		slangi_stats["dataset"],

		wordnetk_stats["dataset"],
		wiktionaryk_stats["dataset"],
		omegak_stats["dataset"],
		
		hei_stats["dataset"],
	]

	syndf = DataFrame(
		ds=String[],
		num_is_syn=Int[],
		pct_is_syn=Float64[],
	)
	for (data, name) in zip(syndfs, syndf_names)
		num_is_syn = get_is_synonyms(filter_single(data))
		pct_is_syn = 100*num_is_syn / length(data.word)

		push!(syndf, (
			name,
			num_is_syn,
			round(pct_is_syn, digits=2),
		))
	end

	syndf
end

# ╔═╡ 660317af-15af-4279-b1c6-4641020977d9
singles_syn_df = sort(
	innerjoin(singles_df, syndf, on=:ds),
	:pct_singles,
	rev=true,
)

# ╔═╡ 2d0f5eb3-a92e-4ac3-ba13-6942fd2f0b13
syn_counts_plot = groupedbar(
	singles_syn_df.ds,
	[singles_syn_df.pct_singles singles_syn_df.pct_is_syn],
	ylabel="Percent (%)",
	xrotation=20,
	label=["Single-word (all)" "Single-word (synoynm)"],
	legend=:topright,
	palette = palette(:Dark2_8),
)

# ╔═╡ cd4b33fc-557b-4837-8d70-2b83000af7a6
md"""
# Outputs
"""

# ╔═╡ 1e9c5088-2b8f-4918-b0e0-d2842249a7b9
md"""
## Plots
"""

# ╔═╡ 975750c2-9130-444e-a619-29e690ac6735
begin
	plot_path = "./plots"
	mkpath(plot_path)
	savefig(polysemes_plot, string(plot_path, "/polys_plot.png"))
	# savefig(overlap_group1_plot, string(plot_path, "/overlap_group1.png"))
	# savefig(overlap_group2_plot, string(plot_path, "/overlap_group2.png"))
	savefig(overlap_all_plot, string(plot_path, "/overlap_all.png"))
	savefig(avg_overlap_plot, string(plot_path, "/avg_overlap.png"))
	# savefig(single_word_plot, string(plot_path, "/single_word.png"))
	# savefig(unique_counts_plot, string(plot_path, "/unique_counts.png"))
	savefig(syn_counts_plot, string(plot_path, "/syn_counts.png"))
end

# ╔═╡ e367363a-886e-4f9e-b117-9cf238e6d6ef
function savetable(df::DataFrame, path::String)
	tex = latexify(df, env=:table, latex=false)
	tex = replace(tex, "%" => "\\%")
	open(path, "w") do io
		write(io, tex)
	end
end

# ╔═╡ cb851950-db0e-4caa-aaa5-03abb0d25951
md"""
## Tables
"""

# ╔═╡ 65bbb80d-d815-4acc-b7d6-a09ad50a51de
defs_table = select(sort(stats_df, :num_definitions, rev=true),
	"dataset" => "Dataset",
	"num_words" => "Words",
	"num_definitions" => "Definitions",
	"definitions_per_word" => "DPW",
	"average_definition_length" => "Mean length",
	"stddev_definition_length" => "SD length",
)

# ╔═╡ db07e864-f146-4977-a8db-fe1ef280f662
polys_table = select(sort(stats_df, :polyseme_ratio, rev=true),
	"dataset" => "Dataset",
	"num_words" => "Words",
	"num_polysemes" => "Polysemes",
	"polyseme_ratio" => "Polysemes (%)",
)

# ╔═╡ 49671ca5-91a6-4697-b112-10342d774ef4
syn_table = select(singles_syn_df,
	"ds" => "Dataset",
	"pct_singles" => "SWD (%)",
	"pct_is_syn" => "SWD synonyms (%)"
)

# ╔═╡ 54073967-751f-47ee-91b1-67214ff8e412
cword_table = select(cword_df,
	"dataset" => "Dataset",
	"definition" => "Definition",
)

# ╔═╡ 7ac14f6e-b594-4a3a-a970-2f28214a4ddb
begin
	table_path = "./tables"
	mkpath(table_path)
	savetable(defs_table, string(table_path, "/defs_table.tex"))
	savetable(polys_table, string(table_path, "/polys_table.tex"))
	savetable(syn_table, string(table_path, "/syn_table.tex"))
	savetable(cword_table, string(table_path, "/cword_table.tex"))
end

# ╔═╡ Cell order:
# ╠═b3fe06ca-a30e-11ec-1de5-4bd280e579c6
# ╠═fb96a818-9397-479d-98d9-02db236c4ee7
# ╠═d2c52b61-27c2-4030-b45a-3888cdadf48c
# ╟─bd240b14-11bf-4c69-957a-b09cefc402e1
# ╠═a98e5705-65fa-46f7-a814-16829d85eb25
# ╠═7a699b59-3c58-4e09-bd62-85ece9fe169c
# ╠═7bf8323c-cb1d-4c0b-a64a-b645359df274
# ╠═f32cc1af-ded4-46c6-b986-87f6e20bcf27
# ╠═a169194a-2da5-4cb6-ac9a-3d9dd0b74f12
# ╠═c30e22da-5f91-410a-8c96-9e6cc7fa00a9
# ╠═7af1a46f-bfa2-44ed-aedc-ac639841882c
# ╠═9d74f007-7856-4a7e-a3ff-175ce69a0bef
# ╠═bf55adef-7f04-473a-b55b-eb7a2edc80f7
# ╟─ddbe8891-5c71-43a8-9009-1a15a8f224b0
# ╠═76048fa5-25f5-44d5-aa82-a9c5c6fa5647
# ╠═918cc3eb-05f9-421e-8770-eb88188128dc
# ╠═a48aad67-cfa9-4d10-a311-add65d2f0781
# ╠═bb334f53-78f4-4d58-9605-90bf9dfe85cd
# ╠═961ca493-f948-40b6-a558-ab6290c07bdd
# ╟─9583cea8-3073-416f-aafd-7366dfc331de
# ╠═e2d6cbdd-9fec-4ecf-881f-9a590a26feb7
# ╠═14ca9b7c-67ad-4d00-9d56-bd528479c0a7
# ╠═df58a7fa-3be6-4266-a185-6da2ce663626
# ╠═1fd2fe2d-f632-46d7-abe7-1f19f5ca55be
# ╟─723796c0-cdd1-498e-8927-4543cd26d1df
# ╠═d94161ca-110b-4eaa-8362-67d0a54068af
# ╠═6a59bd35-d939-4e68-bdab-4fe72cdba7eb
# ╠═e4ab6b4f-ca84-4d31-944f-e98f55aa1917
# ╟─86084ad2-2b4a-4312-bd50-21e1e70c232f
# ╠═7f2b3999-3ed4-4dc2-bf2a-83cafc2539c5
# ╠═80d6fb56-de40-4241-ba3a-a60945a80fe6
# ╠═c48009a0-0a34-4677-bc7a-cb13f1e36201
# ╟─45953cff-df0e-444e-8ac1-96caf8137c99
# ╠═a520f122-57f7-4843-b01d-b80ded800f7d
# ╠═633a9633-86ee-4432-ad19-491a4d7c9ff9
# ╠═06883aaa-98fa-4725-a0fa-f2ba56894955
# ╠═ac998f10-18bc-4a6a-8b68-9e58636d3ced
# ╠═28ef61fd-c23a-4e6c-b9ec-6790ebefe4fc
# ╟─0cc11185-d090-452e-8f91-e6db9bcbce59
# ╠═8b0337d5-9c16-428e-830d-ab116c5eb28c
# ╠═af4fc4a1-b36e-458a-b704-0f3f0628d4fa
# ╠═170dfc16-66bf-4055-a226-f37cfd3ad368
# ╠═6be8b9b0-455f-4aa5-980e-27d19d6cf769
# ╟─35f5c943-5c54-44cd-b6ad-38639904f913
# ╠═55eba273-0377-4de4-8734-db66ca3e2673
# ╠═e9df45c7-44aa-40bf-917f-3cd515687bc0
# ╠═23a8c3a1-ece4-4971-b818-686eb95c7608
# ╟─1d0fb198-1d89-443e-881d-4ecd64673fb8
# ╠═bb18cce0-b79f-4427-a152-ba7561c44966
# ╠═ebb17d85-98c1-4470-bcd0-69e54278ca9f
# ╠═aab6c361-1b94-465e-99b0-fa60d5816e05
# ╟─d2955e46-e745-45ba-a744-531302ca8394
# ╠═f39371ab-77ed-4633-9f37-43c50e01e7ee
# ╠═60d8d452-b6ec-49d2-a138-00003dd0fc4b
# ╟─72cba374-35db-48b7-9077-3bcc27e00ab0
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
# ╟─3f1c8874-260e-47f6-99bd-305c17e55487
# ╟─8eea0d2d-870e-44a5-a2f0-1a3ce5bffd39
# ╠═8a8623b0-70d1-4f69-9804-d2963d18b8ef
# ╠═2a0a3d9c-bbcd-4fd7-b6c6-fe3302a035ef
# ╟─275b9b34-146f-4f9a-be5e-14468841121e
# ╠═741971a7-726c-48a3-81e1-a9653467c3b8
# ╠═57f7faf7-347e-430e-9f4f-6f2c993e826e
# ╟─21bd6d8f-af6f-40a3-a162-61a0c44f8065
# ╠═a600c797-7d09-47c1-8ca3-c872ec19fd78
# ╠═99ae3b96-03e3-45fb-86f9-ce61fc24bd82
# ╟─fee640fe-f35f-4a6f-9e7b-f1315cc93af4
# ╟─a09f51e6-903e-4a12-9ad0-4bd05182d8f8
# ╠═d3be4181-674e-46d9-b39b-9782e42790b0
# ╠═715f81ab-c5fc-4a99-aaab-3e034c0dea43
# ╟─f1ead630-7679-4144-9764-973428111edc
# ╠═a61e7f6c-c553-4494-9c60-10b6543bf586
# ╠═d02bf5b4-9017-4177-8937-0a2eef03bff5
# ╠═6ae940c0-11a1-4d30-a2ea-9779587005ce
# ╟─3fe96326-37eb-4996-b8ea-9557af0b5516
# ╠═a83e980e-f826-4c20-826b-017f067512a6
# ╠═9072c532-9da3-4c11-9cbe-532de868b2e8
# ╟─39de496c-9d01-4a8e-bd98-f74ea45f5ff7
# ╠═2fefd8da-9906-4e8e-826c-761a50b0b633
# ╠═56ee17bb-f197-41a6-98fe-9fd0bef87278
# ╟─f7eb3e18-5217-4b6e-b178-575f04e8b650
# ╟─ae799722-2a64-4716-b2ab-de4fa0072261
# ╠═ff7bac23-36fa-4508-8737-817978d00cec
# ╠═3acf9f9f-777a-4a30-9521-775ddb0fd567
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
# ╟─2ac8b154-2f37-472b-bf79-a73d6f0c37d9
# ╠═dbc8c73e-4d1a-4f64-8d37-5662529aa393
# ╠═d9e23ae2-bf40-46e4-85a4-225912d4cbe4
# ╠═5a8d300e-01fd-4ba7-b6a0-f5186c1291a2
# ╟─bf0afa7a-be35-4c1b-82e3-2817c8d99d91
# ╠═76a4cc6e-969e-4bc4-a193-39b5c5aafba0
# ╠═1b645f58-6f5c-43f4-a759-4b9aff02c921
# ╠═3c38e58f-c9b8-41ed-bbdb-73738295aeaf
# ╟─5329894b-0c64-4d6d-b8cd-5214e938205a
# ╠═ca87b2dc-621a-4431-bff1-834d16314004
# ╠═8c15216b-17b4-43b0-821a-57cf602c3c4e
# ╠═7f3ef8e7-dd72-4645-8048-968f596ed73e
# ╟─eb751ce5-df60-46b5-9545-97cd13ad9a95
# ╠═91dff8f9-e826-465b-9cd4-2666b2c972cf
# ╠═02a18190-3955-463a-8bad-fd7f92257075
# ╠═8b098243-2467-40fb-9836-3750008312d1
# ╠═4b351ddc-0927-4128-aaf4-43794a845762
# ╠═9dd70ccb-d434-440d-8da5-f1caf3c0dd5d
# ╠═6654623f-0ae9-4c5f-8820-5957b87f0225
# ╠═310b2b16-7c2d-43c1-a47a-17b0f1e58bad
# ╠═3eba3f16-d7d9-4af2-907d-38c445d7a978
# ╠═d5f4a207-a5dc-4636-96fd-d6eb37a16c8d
# ╟─ac35abd6-2e14-40bc-9f08-0b82d7b33ee1
# ╠═b4cb80f5-ff4b-4b87-8442-bb57e0ba0884
# ╠═75325322-d3b1-4fac-85ca-dc9c5b3f4a7a
# ╠═9f5ff836-df46-4214-8212-22edd21054d8
# ╠═ffa82e4d-71f0-4066-8f64-1178d35ca285
# ╠═0effd27e-d000-4aad-b7db-1dbebdc2dceb
# ╠═f685fb61-9b86-4a2b-93bc-41d5f8bba763
# ╟─087f2cde-9fe6-4255-9ffd-b4aafeeee2f4
# ╠═834789f8-7e31-4bfb-a738-9fd9d9b2d5b3
# ╠═b3f9193f-6e73-4824-9269-1fd65c115750
# ╠═54e23520-4348-42ab-8545-59475005dbe8
# ╠═ae280702-b9aa-41c8-a597-125dacad1bab
# ╠═5df06c9e-1fdd-455f-9330-a44761824eb1
# ╠═dffd4b78-b502-4a69-b0c3-6961cca9632e
# ╠═660317af-15af-4279-b1c6-4641020977d9
# ╠═2d0f5eb3-a92e-4ac3-ba13-6942fd2f0b13
# ╟─cd4b33fc-557b-4837-8d70-2b83000af7a6
# ╟─1e9c5088-2b8f-4918-b0e0-d2842249a7b9
# ╠═975750c2-9130-444e-a619-29e690ac6735
# ╠═e367363a-886e-4f9e-b117-9cf238e6d6ef
# ╟─cb851950-db0e-4caa-aaa5-03abb0d25951
# ╠═65bbb80d-d815-4acc-b7d6-a09ad50a51de
# ╠═db07e864-f146-4977-a8db-fe1ef280f662
# ╠═49671ca5-91a6-4697-b112-10342d774ef4
# ╠═54073967-751f-47ee-91b1-67214ff8e412
# ╠═7ac14f6e-b594-4a3a-a970-2f28214a4ddb
