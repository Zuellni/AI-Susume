class Model {
	constructor(shape) {
		this.model = tf.sequential()
		this.model.add(tf.layers.batchNormalization({ inputShape: [shape] }))
		this.model.add(tf.layers.dense({ units: 256, activation: "relu", kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) }))
		this.model.add(tf.layers.dropout({ rate: 0.3 }))
		this.model.add(tf.layers.dense({ units: 64, activation: "relu", kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) }))
		this.model.add(tf.layers.dropout({ rate: 0.3 }))
		this.model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }))
		this.model.compile({ loss: "meanSquaredError", metrics: "mse", optimizer: "adam" })
	}

	async train(data, epochs, patience, callbacks) {
		const inputs = data.map(item => item.input)
		const outputs = data.map(item => item.score)
		const inputsTensor = tf.tensor(inputs)
		const outputsTensor = tf.tensor(outputs)

		await this.model.fit(inputsTensor, outputsTensor, {
			callbacks: [
				new tf.CustomCallback(callbacks),
				tf.callbacks.earlyStopping({ monitor: "val_loss", patience: patience }),
			],
			batchSize: inputs.length,
			epochs: epochs,
			shuffle: true,
			validationSplit: 0.2,
		})

		tf.dispose([inputsTensor, outputsTensor])
	}

	async predict(data, limit) {
		const inputs = data.map(item => item.input)
		const inputsTensor = tf.tensor(inputs)
		const predictions = this.model.predict(inputsTensor, { batchSize: inputs.length })
		const predictionsData = await predictions.data()

		tf.dispose([inputsTensor, predictions])

		const outputs = data.map((item, index) => ({
			score: predictionsData[index],
			siteUrl: item.siteUrl,
			title: item.title,
		}))

		return outputs.sort((a, b) => b.score - a.score).slice(0, limit)
	}
}

const stringSimilarity = (first, second) => {
	const stack = [first, second]
	let score = 0

	while (stack.length !== 0) {
		const fss = stack.pop()
		const sss = stack.pop()
		let lsl = 0
		let flsi = -1
		let slsi = -1

		for (let i = 0; i < fss.length; i++) {
			for (let j = 0; j < sss.length; j++) {
				let k = 0

				while (i + k < fss.length && j + k < sss.length && fss.charAt(i + k) === sss.charAt(j + k))
					k++

				if (k > lsl) {
					lsl = k
					flsi = i
					slsi = j
				}
			}
		}

		if (lsl > 0) {
			score += lsl * 2

			if (flsi !== 0 && slsi !== 0) {
				stack.push(fss.substring(0, flsi))
				stack.push(sss.substring(0, slsi))
			}

			if (flsi + lsl !== fss.length && slsi + lsl !== sss.length) {
				stack.push(fss.substring(flsi + lsl, fss.length))
				stack.push(sss.substring(slsi + lsl, sss.length))
			}
		}
	}

	return score / (first.length + second.length)
}

const getList = async (user, type, token) => {
	const options = {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"Accept": "application/json",
			...(token && { "Authorization": `Bearer ${token}` }),
		},
		body: JSON.stringify({
			query: `query ($user: String, $type: MediaType) {
				MediaListCollection(userName: $user, type: $type) {
					lists { entries {
							score (format: POINT_100)
							media { siteUrl }
						}
					}
				}
			}`,
			variables: {
				user: user,
				type: type,
			},
		}),
	}

	const response = await fetch("https://graphql.anilist.co", options)
	const json = response.ok && await response.json()
	const lists = json.data.MediaListCollection.lists
	const entries = {}
	let min = Infinity
	let max = -Infinity

	for (const list of lists)
		for (const entry of list.entries)
			if (entry.score) {
				entries[entry.media.siteUrl] = entry.score

				if (entry.score < min)
					min = entry.score

				if (entry.score > max)
					max = entry.score
			}

	return { entries: entries, min: min, max: max }
}

const loadDatabase = async () => {
	const response = await fetch("https://raw.githubusercontent.com/manami-project/anime-offline-database/master/anime-offline-database-minified.json")
	const json = response.ok && await response.json()
	const data = json.data

	const tags = new Set(data.flatMap(entry => entry.tags.map(tag => tag.toUpperCase())))
	const tagGroups = []
	const tagMapping = {}
	const threshold = 0.7

	for (const tag of tags) {
		let added = false

		for (const group of tagGroups)
			if (stringSimilarity(tag, group[0]) >= threshold) {
				group.push(tag)
				added = true
				break
			}

		if (!added)
			tagGroups.push([tag])
	}

	for (const group of tagGroups) {
		const representativeTag = group[0]

		for (const tag of group)
			tagMapping[tag] = representativeTag
	}

	const updatedData = data.map(entry => ({
		...entry,
		tags: entry.tags.map(tag => tagMapping[tag.toUpperCase()] || tag.toUpperCase())
	}))

	const finalTags = [...new Set(updatedData.flatMap(entry => entry.tags))].sort()
	return { data: updatedData, schema: finalTags }
}

const prepareData = (entries, database) => {
	const schema = database.schema
	const schemaLen = schema.length
	const data = database.data
	const dataTrain = []
	const dataPredict = []

	for (const { animeSeason, sources, status, tags, title, type } of data) {
		const siteUrl = sources.find(item => item.startsWith("https://anilist.co"))

		if (!animeSeason.year || !siteUrl || type === "UNKNOWN")
			continue

		let input = new Array(schemaLen).fill(0)
		const score = entries[siteUrl]

		for (const tag of tags) {
			const index = schema.indexOf(tag.toUpperCase())

			if (index !== -1)
				input[index] = 1
		}

		if (input.some(value => value == 1)) {
			if (score)
				dataTrain.push({
					input: input,
					score: score,
				})
			else if (status === "FINISHED" || status === "ONGOING")
				dataPredict.push({
					input: input,
					siteUrl: siteUrl,
					title: title,
					type: type,
					year: animeSeason.year,
				})
		}
	}

	return { train: dataTrain, predict: dataPredict }
}

const normalize = (obj, oldMin, oldMax, newMin, newMax) => {
	return Object.fromEntries(
		Object.entries(obj).map(([key, value]) => [
			key,
			(value - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin,
		])
	)
}

const elements = document.querySelectorAll("input, select")
const enable = (elements) => elements.forEach(element => element.disabled = false)
const disable = (elements) => elements.forEach(element => element.disabled = true)

const user = document.querySelector("#user")
const token = document.querySelector("#token")
const get = document.querySelector("#get")
const getValue = get.value

const epochs = document.querySelector("#epochs")
const patience = document.querySelector("#patience")
const train = document.querySelector("#train")
const trainValue = train.value

const yearFrom = document.querySelector("#yearFrom")
const yearTo = document.querySelector("#yearTo")
const year = new Date().getFullYear()

yearFrom.max = year
yearTo.value = year
yearTo.max = year

const type = document.querySelector("#type")
const limit = document.querySelector("#limit")
const predict = document.querySelector("#predict")
const predictValue = predict.value

const main = document.querySelector("main")
let list, data, model

get.addEventListener("click", async () => {
	disable(elements)
	get.value = "Getting List..."

	try {
		list = await getList(user.value, "ANIME", token.value)
		list.entries = normalize(list.entries, list.min, list.max, 0, 1)

		get.value = getValue
		console.log(list)
	} catch (e) {
		get.value = "Error"
		console.log(e)
	}

	enable(elements)
})

train.addEventListener("click", async () => {
	disable(elements)
	train.value = "Training Model..."

	const callbacks = {
		onEpochEnd: (epoch, log) => {
			epoch = epoch / epochs.value * 100
			train.style.background = `linear-gradient(90deg, var(--acd), ${epoch}%, var(--acl), ${epoch}%, var(--acl))`
			console.log(`loss: ${log.loss},\tval_loss: ${log.val_loss}`)
		},
		onTrainEnd: () => train.removeAttribute("style")
	}

	try {
		const database = await loadDatabase()
		console.log(database)

		data = prepareData(list.entries, database)
		console.log(data)

		model = new Model(database.schema.length)
		await model.train(data.train, epochs.value, patience.value, callbacks)
		train.value = trainValue
	} catch (e) {
		train.value = "Error"
		console.log(e)
	}

	enable(elements)
})

predict.addEventListener("click", async () => {
	disable(elements)
	predict.value = "Recommending..."
	main.replaceChildren()

	const dataPredict = data.predict.filter(item =>
		item.type === type.value
		&& item.year >= yearFrom.value
		&& item.year <= yearTo.value
	)

	let predictions, scores, scoresNorm

	try {
		predictions = await model.predict(dataPredict, limit.value)
		scores = predictions.map(pred => pred.score)

		let min = Math.min(...scores)
		let max = Math.max(...scores)

		min = min < 0 ? min : 0
		max = max > 1 ? max : 1

		scoresNorm = normalize(scores, min, max, list.min, list.max)

		predict.value = predictValue
		console.log(predictions)
	} catch (e) {
		predict.value = "Error"
		console.log(e)
	}

	if (predictions && scores && scoresNorm) {
		const header = document.createElement("output")
		const title = document.createElement("span")
		const score = document.createElement("span")

		title.innerText = "Title"
		score.innerText = "Score"

		header.appendChild(title)
		header.appendChild(score)
		main.appendChild(header)

		for (let i = 0; i < predictions.length; i++) {
			const row = document.createElement("output")
			const link = document.createElement("a")
			const span = document.createElement("span")
			const score = document.createElement("span")

			link.href = predictions[i].siteUrl
			link.target = "_blank"
			link.innerText = predictions[i].title
			score.innerText = Math.round(scoresNorm[i])

			span.appendChild(link)
			row.appendChild(span)
			row.appendChild(score)
			main.appendChild(row)
		}
	}

	enable(elements)
})