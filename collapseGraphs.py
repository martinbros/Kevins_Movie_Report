import pretty_errors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
import webbrowser
from datetime import datetime
import math

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams["figure.figsize"] = (16,9)


def custom_round(x, base=5):
	return int(base * math.floor(float(x) / base))


def getRoundedThresholdv1(a, MinClip):
	return round(float(a) / MinClip) * MinClip


def genHist(df, ident, sortIdent, sortBool, coordName):
	df = df[[ident, "Watch Date"]]  # Reduce Dataframe, the order is important
	df = df.explode(ident)  # Duplicate the movie per the number of writers

	# The following lines regroup the movies to the person who made it
	keys, values = df.sort_values(ident).values.T
	ukeys, index = np.unique(keys, True)
	arrays = np.split(values, index[1:])
	df2 = pd.DataFrame({ident: ukeys, "Watch Date": [list(a) for a in arrays]})

	df2.set_index(keys=[ident], inplace=True)

	df2["len"] = df2["Watch Date"].str.len()  # per ident, get how may pictures there are
	df2.sort_values(by=[sortIdent], ascending=sortBool, inplace=True)  # Reorder the dataframe
	df2[coordName] = range(1, df2.shape[0] + 1)

	numColorCycle = int(df2.shape[0] / len(colors)) + (df2.shape[0] % len(colors) > 0)  # How many times to repeat the cycle
	cycledColors = colors * numColorCycle  # Creating the repeated cycle
	df2["color"] = cycledColors[:df2.shape[0]]  # Creating the color column

	return df2


#Big Dots
def mashTwoHist(yHist, yIdent, xHist, xIdent):
	dataIdents = [xIdent, yIdent, "Long Title", "IMDB URL (don't edit me)", "runtimes"]

	if "genres" in dataIdents:
		graphingData = smDF[dataIdents].explode("genres")  # Duplicate entries with multiples genres
	else:
		graphingData = smDF[dataIdents]

	graphingData = pd.merge(graphingData, yHist[["yCord"]], on=yIdent)  # Create "yCord" column for the appropriate y coordinate
	graphingData = pd.merge(graphingData, xHist[["xCord", "color"]], on=xIdent)  # Create "xCord" column for the appropriate x coordinate
	graphingData["coord"] = graphingData[["xCord", "yCord"]].apply(lambda x: ",".join(x.astype(str)), axis=1)  # Create coordinate column

	if "runtimes15" in dataIdents:
		graphingData["annot"] = graphingData["Long Title"] + " : " + graphingData["runtimes"].astype(str) + " min."
	else:
		graphingData["annot"] = graphingData.loc[:, "Long Title"]

	coordDataLists = {}
	for ident in [xIdent, "annot", "IMDB URL (don't edit me)", "Long Title"]:
		coordDataLists[ident] = graphingData.groupby('coord')[ident].apply(list).reset_index(name=ident).set_index('coord')  # For a coordinate, get all of the data collapsed into a list

	mashedDF = pd.concat(coordDataLists.values(), axis=1)  # Combine all of the DFs according to the coordinate
	mashedDF[xIdent] = mashedDF[xIdent].str[0]  # Collapse list of xIdent into a single string
	mashedDF["movieCount"] = mashedDF["Long Title"].str.len()  # Get the count of movies at this coordinate
	mashedDF.reset_index(inplace=True)

	mashedDF[['xCord', 'yCord']] = mashedDF["coord"].str.split(',', n=1, expand=True).astype(int)
	mashedDF = pd.merge(mashedDF, xHist[["len", "color"]], on=xIdent)  # Get the scalar at which to size the point
	mashedDF["size"] = mashedDF["movieCount"] / mashedDF["len"] * 5000  # Create the size of the point

	graphInfo = {}
	graphInfo["x"] = list(mashedDF["xCord"])
	graphInfo["y"] = list(mashedDF["yCord"])
	graphInfo["c"] = list(mashedDF["color"])
	graphInfo["s"] = list(mashedDF["size"])
	graphInfo["count"] = list(mashedDF["movieCount"])
	graphInfo["title"] = list(mashedDF["annot"])
	graphInfo["hyperlink"] = list(mashedDF["IMDB URL (don't edit me)"])

	return graphInfo


#Timeline
def mashHistAndTimeline(hist, ident):

	if ident == "genres":
		graphingData = smDF[[ident, "Long Title", "Watch Date", "IMDB URL (don't edit me)"]].explode(ident)  # Duplicate entries with multiples genres
	else:
		graphingData = smDF[[ident, "Long Title", "Watch Date", "IMDB URL (don't edit me)"]]
	graphingData = pd.merge(graphingData, hist[["yCord", "color"]], on=ident)  # Create "yTicks" column for the appropriate y coordinate
	graphingData['Watch Date'] = pd.to_datetime(graphingData['Watch Date'], format="mixed")
	graphingData["annot"] = graphingData["Long Title"] + " : " + graphingData["Watch Date"].dt.strftime("%d %b")
	graphingData["size"] = 50
	graphingData["movieCount"] = None

	graphInfo = {}
	graphInfo["x"] = list(graphingData["Watch Date"])
	graphInfo["y"] = list(graphingData["yCord"])
	graphInfo["c"] = list(graphingData["color"])
	graphInfo["s"] = list(graphingData["size"])
	graphInfo["count"] = list(graphingData["movieCount"])
	graphInfo["title"] = list(graphingData["annot"])
	graphInfo["hyperlink"] = list(graphingData["IMDB URL (don't edit me)"])

	return graphInfo


def genInteractableScatter(graphInfo, fig, ax):
	sc = plt.scatter(graphInfo["x"], graphInfo["y"], c=graphInfo["c"], s=graphInfo["s"])

	yMax = max(graphInfo["y"])
	numNamesPerDiv = -0.3 * yMax + 8.0
	xRange = max(graphInfo["x"]) - min(graphInfo["x"])
	xMax = (xRange * 0.8) + min(graphInfo["x"])

	for idx, point in enumerate(graphInfo["x"]):
		ax.text(graphInfo["x"][idx], graphInfo["y"][idx], graphInfo["count"][idx], ha="center", va="center")

	annot = ax.annotate("", xy=(0, 0), xytext=(20, 10), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->", color="k"))
	annot.set_visible(False)

	def update_annot(ind):

		if yMax < 50:
			verticalSpace = (yMax - graphInfo["y"][ind["ind"][0]]) * numNamesPerDiv
			verticalSpace = int(verticalSpace)
		else:
			verticalSpace = 50

		if isinstance(graphInfo["title"][ind["ind"][0]], str):
			text = ["{}".format(graphInfo["title"][n]) for n in ind["ind"]]

			numEntries = len(text)
			text = "\n".join(text)

		if isinstance(graphInfo["title"][ind["ind"][0]], list):
			numEntries = len(graphInfo["title"][ind["ind"][0]])
			text = "{}".format("\n".join(graphInfo["title"][ind["ind"][0]]))

		pos = sc.get_offsets()[ind["ind"][0]]
		annot.xy = pos

		if verticalSpace < numEntries:
			annot._y = (numEntries - verticalSpace) * -10
		else:
			annot._y = 10

		if graphInfo["x"][ind["ind"][0]] > xMax:
			annot._horizontalalignment = "right"
		else:
			annot._horizontalalignment = "left"

		annot.set_text(text)
		annot.get_bbox_patch().set_facecolor(graphInfo["c"][ind["ind"][0]])
		annot.get_bbox_patch().set_alpha(0.8)
		#attrs=vars(annot)
		#print(', '.join("%s: %s\n" % item for item in attrs.items()))

	def hover(event):
		vis = annot.get_visible()
		if event.inaxes == ax:
			cont, ind = sc.contains(event)
			if cont:
				update_annot(ind)
				annot.set_visible(True)
				fig.canvas.draw_idle()
			else:
				if vis:
					annot.set_visible(False)
					fig.canvas.draw_idle()

	fig.canvas.mpl_connect("motion_notify_event", hover)

	def onclick(event):
		if event.inaxes == ax:
			cont, ind = sc.contains(event)
			if cont:
				if isinstance(graphInfo["hyperlink"][ind["ind"][0]], list):
					for link in graphInfo["hyperlink"][ind["ind"][0]]:
						webbrowser.open(link)
				else:
					links = ["{}".format(graphInfo["hyperlink"][n]) for n in ind["ind"]]
					for link in links:
						webbrowser.open(link)

	fig.canvas.mpl_connect('button_press_event', onclick)


smDF = pd.read_pickle("movieDataset.pkl")
smDF["runtimes15"] = pd.Series(smDF["runtimes"]).apply(lambda x: getRoundedThresholdv1(x, 15))

#################
#
# Generating Histograms
#
#################
histRating = genHist(smDF, "kevRatingSingle", "kevRatingSingle", True, "yCord")
histRating["xCord"] = histRating.loc[:, "yCord"]
histRating["axisTicks"] = histRating.index.astype(str) + "/10 [" + histRating.len.astype(str) + " movies] "

histGenre = genHist(smDF, "genres", "len", False, "xCord")
histGenre["yCord"] = histGenre.loc[:, "xCord"]
histGenre["axisTicks"] = histGenre.index + "[" + histGenre.len.astype(str) + "]"

smDF["decade"] = pd.Series(smDF["year"]).apply(lambda x: custom_round(x, base=10))
histDecade = genHist(smDF, "decade", "len", False, "yCord")
histDecade["xCord"] = histDecade.loc[:, "yCord"]
histDecade["axisTicks"] = histDecade.index.astype(str) + "[" + histDecade.len.astype(str) + "]"

histRuntime = genHist(smDF, "runtimes15", "runtimes15", True, "yCord")
histRuntime["xCord"] = histRuntime.loc[:, "yCord"]
histRuntime["axisTicks"] = histRuntime.index.astype(str) + " min. " + "[" + histRuntime.len.astype(str) + "]"

#################
#
# Graphing Rating vs Timeline
#
#################
graphInfoRatVsTim = mashHistAndTimeline(histRating, "kevRatingSingle")

figRatVsTim, axRatVsTim = plt.subplots()
#attrs=vars(figRatVsTim)
#print(', '.join("%s: %s\n" % item for item in attrs.items()))
genInteractableScatter(graphInfoRatVsTim, figRatVsTim, axRatVsTim)

axRatVsTim.set_yticks(histRating["yCord"], histRating.axisTicks)
for ytick, color in zip(axRatVsTim.get_yticklabels(), list(histRating["color"])):
	ytick.set_color(color)

axRatVsTim.xaxis.set_major_locator(mdates.MonthLocator())
axRatVsTim.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

axRatVsTim.set_title("Kevin's Rating vs Watch Date\n$^\mathrm{{Click\ a\ Dot!}}$")
axRatVsTim.set_xlabel("Watch Date")
axRatVsTim.set_ylabel("Kevin's Rating [Total Watched]")

#figRatVsTim.tight_layout()

#################
#
# Graphing Genre vs Timeline
#
#################
graphInfoGenVsTim = mashHistAndTimeline(histGenre, "genres")

figGenVsTim, axGenVsTim = plt.subplots()
genInteractableScatter(graphInfoGenVsTim, figGenVsTim, axGenVsTim)

axGenVsTim.set_yticks(histGenre["yCord"], histGenre.axisTicks)
for ytick, color in zip(axGenVsTim.get_yticklabels(), list(histGenre["color"])):
	ytick.set_color(color)

axGenVsTim.xaxis.set_major_locator(mdates.MonthLocator())
axGenVsTim.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

axGenVsTim.set_title("Movie Genre vs Watch Date\n$^\mathrm{{Click\ a\ Dot!}}$")
axGenVsTim.set_xlabel("Watch Date")
axGenVsTim.set_ylabel("Movie Genre [Total Watched]")

#figGenVsTim.tight_layout()

#################
#
# Graphing Decade vs Timeline
#
#################
graphInfoDecVsTim = mashHistAndTimeline(histDecade, "decade")

figDecVsTim, axDecVsTim = plt.subplots()
genInteractableScatter(graphInfoDecVsTim, figDecVsTim, axDecVsTim)

axDecVsTim.set_yticks(histDecade["yCord"], histDecade.axisTicks)
for ytick, color in zip(axDecVsTim.get_yticklabels(), list(histDecade["color"])):
	ytick.set_color(color)

axDecVsTim.xaxis.set_major_locator(mdates.MonthLocator())
axDecVsTim.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

axDecVsTim.set_title("Decade vs Watch Date\n$^\mathrm{{Click\ a\ Dot!}}$")
axDecVsTim.set_xlabel("Watch Date")
axDecVsTim.set_ylabel("Decade [Total Watched]")

#figDecVsTim.tight_layout()

#################
#
# Graphing Runtime vs Timeline
#
#################
smDF['datetime'] = pd.to_datetime(smDF['Watch Date'], format="mixed")  # Create datetime column
smDF = smDF.merge(histRuntime[["color"]], how="left", on="runtimes15")  # Assign Colors
smDF["size"] = 50
smDF["movieCount"] = None
smDF["annot"] = smDF["Long Title"] + " : " + smDF["datetime"].dt.strftime("%d %b") + " : " + smDF["runtimes"].astype(str) + " min."


graphInfoRunVsTim = {}
graphInfoRunVsTim["x"] = list(smDF["datetime"])
graphInfoRunVsTim["y"] = list(smDF["runtimes"])
graphInfoRunVsTim["c"] = list(smDF["color"])
graphInfoRunVsTim["s"] = list(smDF["size"])
graphInfoRunVsTim["count"] = list(smDF["movieCount"])
graphInfoRunVsTim["title"] = list(smDF["annot"])
graphInfoRunVsTim["hyperlink"] = list(smDF["IMDB URL (don't edit me)"])

figRunVsTim, axRunVsTim = plt.subplots()
genInteractableScatter(graphInfoRunVsTim, figRunVsTim, axRunVsTim)

axRunVsTim.xaxis.set_major_locator(mdates.MonthLocator())
axRunVsTim.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

loc = plticker.MultipleLocator(base=15)  # this locator puts ticks at regular intervals
axRunVsTim.yaxis.set_major_locator(loc)

axRunVsTim.set_title("Movie Runtime vs Watch Date\n$^\mathrm{{Click\ a\ Dot!}}$")
axRunVsTim.set_xlabel("Watch Date")
axRunVsTim.set_ylabel("Movie Runtime (minutes)")

#figRunVsTim.tight_layout()

#################
# Big Dots
# Graphing Rating vs Genre
#
#################
graphInfoRatVsGen = mashTwoHist(histRating, "kevRatingSingle", histGenre, "genres")

figRatVsGen, axRatVsGen = plt.subplots()
genInteractableScatter(graphInfoRatVsGen, figRatVsGen, axRatVsGen)

axRatVsGen.set_yticks(histRating["yCord"], histRating.axisTicks)
axRatVsGen.set_xticks(histGenre["xCord"], histGenre.axisTicks, rotation=20, ha="right")
for xtick, color in zip(axRatVsGen.get_xticklabels(), list(histGenre["color"])):
	xtick.set_color(color)

axRatVsGen.axis('equal')
axRatVsGen.set_title("Kevin's Rating vs Genre\n$^\mathrm{{Click\ a\ Dot!}}$\n$^\mathrm{{The\ size\ of\ the\ dot\ reflects\ the\ number\ of\ movies\ at\ that\ specific\ coordinate}}$\n$^\mathrm{{All\ circles\ are\ scaled\ to\ the\ number\ of\ movies\ watched\ per\ genre}}$")
axRatVsGen.set_xlabel("Movie Genre [Total Watched]")
axRatVsGen.set_ylabel("Kevin's Rating [Total Watched]")

#figRatVsGen.tight_layout()

#################
# Big Dots
# Graphing Runtime vs Genre
#
#################
graphInfoRunVsGen = mashTwoHist(histRuntime, "runtimes15", histGenre, "genres")

figRunVsGen, axRunVsGen = plt.subplots()
genInteractableScatter(graphInfoRunVsGen, figRunVsGen, axRunVsGen)

axRunVsGen.set_yticks(histRuntime["yCord"], histRuntime.axisTicks)
axRunVsGen.set_xticks(histGenre["xCord"], histGenre.axisTicks, rotation=20, ha="right")
for xtick, color in zip(axRunVsGen.get_xticklabels(), list(histGenre["color"])):
	xtick.set_color(color)

axRunVsGen.axis('equal')
axRunVsGen.set_title("Movie Runtime vs Genre\n$^\mathrm{{Click\ a\ Dot!}}$\n$^\mathrm{{The\ size\ of\ the\ dot\ reflects\ the\ number\ of\ movies\ at\ that\ specific\ coordinate}}$\n$^\mathrm{{All\ circles\ are\ scaled\ to\ the\ number\ of\ movies\ watched\ per\ genre}}$")
axRunVsGen.set_xlabel("Movie Genre [Total Watched]")
axRunVsGen.set_ylabel("Movie Runtime (rounded to closest 15 min.) [Total Watched]")

#figRunVsGen.tight_layout()

#################
# Big Dots
# Graphing Decade vs Genre
#
#################
graphInfoDecVsGen = mashTwoHist(histDecade, "decade", histGenre, "genres")

figDecVsGen, axDecVsGen = plt.subplots()
genInteractableScatter(graphInfoDecVsGen, figDecVsGen, axDecVsGen)

axDecVsGen.set_yticks(histDecade["yCord"], histDecade.axisTicks)
axDecVsGen.set_xticks(histGenre["xCord"], histGenre.axisTicks, rotation=20, ha="right")
for xtick, color in zip(axDecVsGen.get_xticklabels(), list(histGenre["color"])):
	xtick.set_color(color)

axDecVsGen.axis('equal')
axDecVsGen.set_title("Decade vs Genre\n$^\mathrm{{Click\ a\ Dot!}}$\n$^\mathrm{{The\ size\ of\ the\ dot\ reflects\ the\ number\ of\ movies\ at\ that\ specific\ coordinate}}$\n$^\mathrm{{All\ circles\ are\ scaled\ to\ the\ number\ of\ movies\ watched\ per\ genre}}$")
axDecVsGen.set_xlabel("Movie Genre [Total Watched]")
axDecVsGen.set_ylabel("Decade [Total Watched]")

#figDecVsGen.tight_layout()

#################
# Big Dots
# Graphing Decade vs Runtime
#
#################
graphInfoDecVsRun = mashTwoHist(histDecade, "decade", histRuntime, "runtimes15")

figDecVsRun, axDecVsRun = plt.subplots()
genInteractableScatter(graphInfoDecVsRun, figDecVsRun, axDecVsRun)

axDecVsRun.set_yticks(histDecade["yCord"], histDecade.axisTicks)
axDecVsRun.set_xticks(histRuntime["xCord"], histRuntime.axisTicks, rotation=20, ha="right")
for xtick, color in zip(axDecVsRun.get_xticklabels(), list(histRuntime["color"])):
	xtick.set_color(color)

axDecVsRun.set_title("Decade vs Runtime\n$^\mathrm{{Click\ a\ Dot!}}$\n$^\mathrm{{The\ size\ of\ the\ dot\ reflects\ the\ number\ of\ movies\ at\ that\ specific\ coordinate}}$\n$^\mathrm{{All\ circles\ are\ scaled\ to\ the\ number\ of\ movies\ watched\ per\ movie\ length}}$")
axDecVsRun.set_xlabel("Movie Runtime (rounded to closest 15 min.) [Total Watched]")
axDecVsRun.set_ylabel("Decade [Total Watched]")

#figDecVsRun.tight_layout()

#################
# Big Dots
# Graphing Rating vs Runtime
#
#################
graphInfoRatVsRun = mashTwoHist(histRating, "kevRatingSingle", histRuntime, "runtimes15")

figRatVsRun, axRatVsRun = plt.subplots()
genInteractableScatter(graphInfoRatVsRun, figRatVsRun, axRatVsRun)

axRatVsRun.set_yticks(histRating["yCord"], histRating.axisTicks)
axRatVsRun.set_xticks(histRuntime["xCord"], histRuntime.axisTicks, rotation=20, ha="right")
for xtick, color in zip(axRatVsRun.get_xticklabels(), list(histRuntime["color"])):
	xtick.set_color(color)

axRatVsRun.set_title("Kevin's Rating vs Runtime\n$^\mathrm{{Click\ a\ Dot!}}$\n$^\mathrm{{The\ size\ of\ the\ dot\ reflects\ the\ number\ of\ movies\ at\ that\ specific\ coordinate}}$\n$^\mathrm{{All\ circles\ are\ scaled\ to\ the\ number\ of\ movies\ watched\ per\ movie\ length}}$")
axRatVsRun.set_xlabel("Movie Runtime (rounded to closest 15 min.) [Total Watched]")
axRatVsRun.set_ylabel("Kevin's Rating [Total Watched]")

#figRatVsRun.tight_layout()

#################
# Big Dots
# Graphing Rating vs Decade
#
#################
graphInfoRatVsRun = mashTwoHist(histRating, "kevRatingSingle", histDecade, "decade")

figRatVsRun, axRatVsRun = plt.subplots()
genInteractableScatter(graphInfoRatVsRun, figRatVsRun, axRatVsRun)

axRatVsRun.set_yticks(histRating["yCord"], histRating.axisTicks)
axRatVsRun.set_xticks(histDecade["xCord"], histDecade.axisTicks, rotation=20, ha="right")
for xtick, color in zip(axRatVsRun.get_xticklabels(), list(histDecade["color"])):
	xtick.set_color(color)

axRatVsRun.set_title("Kevin's Rating vs Decade\n$^\mathrm{{Click\ a\ Dot!}}$\n$^\mathrm{{The\ size\ of\ the\ dot\ reflects\ the\ number\ of\ movies\ at\ that\ specific\ coordinate}}$\n$^\mathrm{{All\ circles\ are\ scaled\ to\ the\ number\ of\ movies\ watched\ per\ decade}}$")
axRatVsRun.set_xlabel("Decade [Total Watched]")
axRatVsRun.set_ylabel("Kevin's Rating [Total Watched]")

#figRatVsRun.tight_layout()

plt.show()