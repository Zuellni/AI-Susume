class Model {
	constructor(shape) {
		this.model = tf.sequential()
		this.model.add(tf.layers.dense({ units: 512, activation: "relu", inputShape: [shape] }))
		this.model.add(tf.layers.dense({ units: 128, activation: "relu" }))
		this.model.add(tf.layers.dense({ units: 32, activation: "relu" }))
		this.model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }))
		this.model.compile({ loss: "meanSquaredError", optimizer: "adam", metrics: "acc" })
	}

	async train(data, epochs, patience, callbacks) {
		const inputs = data.map(item => item.input)
		const outputs = data.map(item => item.score)
		const inputsTensor = tf.tensor2d(inputs)
		const outputsTensor = tf.tensor1d(outputs)
		const batchSize = inputs.length

		await this.model.fit(inputsTensor, outputsTensor, {
			batchSize: batchSize,
			callbacks: [
				new tf.CustomCallback(callbacks),
				tf.callbacks.earlyStopping({ monitor: "acc", patience: patience }),
			],
			epochs: epochs,
		})

		tf.dispose([inputsTensor, outputsTensor])
	}

	async predict(data, limit) {
		const inputs = data.map(item => item.input)
		const inputsTensor = tf.tensor2d(inputs)
		const preds = this.model.predict(inputsTensor)
		const predsData = await preds.data()
		tf.dispose([inputsTensor, preds])

		const outputs = data.map((item, index) => ({
			score: predsData[index],
			siteUrl: item.siteUrl,
			title: item.title,
		}))

		return outputs.sort((a, b) => b.score - a.score).slice(0, limit)
	}
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
	return json.data
}

const createSchema = (database, types) => {
	const schema = new Set()

	for (const { tags } of database)
		for (const tag of tags)
			schema.add(tag.toUpperCase())

	for (const type of types)
		schema.add(type)

	return Array.from(schema)
}

const prepareData = (entries, database, schema) => {
	const schemaLen = schema.length
	const dataTrain = []
	const dataPred = []

	for (const { animeSeason, sources, status, tags, title, type } of database) {
		const siteUrl = sources.find(item => item.startsWith("https://anilist.co"))

		if (!animeSeason.year || !siteUrl || type == "UNKNOWN")
			continue

		const input = new Array(schemaLen).fill(0)
		const score = entries[siteUrl]
		const index = schema.indexOf(type)

		if (index != -1)
			input[index] = 1

		for (const tag of tags) {
			const index = schema.indexOf(tag.toUpperCase())

			if (index != -1)
				input[index] = 1
		}

		if (score)
			dataTrain.push({
				input: input,
				score: score,
			})
		else if (status == "FINISHED" || status == "ONGOING")
			dataPred.push({
				input: input,
				siteUrl: siteUrl,
				title: title,
				type: type,
				year: animeSeason.year,
			})
	}

	return { train: dataTrain, pred: dataPred }
}

const normalize = (obj, oldMin, oldMax, newMin, newMax) => {
	return Object.fromEntries(
		Object.entries(obj).map(([key, value]) => [
			key,
			(value - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin,
		])
	)
}

const disable = (elements) => {
	for (const element of elements)
		element.disabled = true
}

const enable = (elements) => {
	for (const element of elements)
		element.disabled = false
}

const elements = document.querySelectorAll("input, select")

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
let list, model, data

get.addEventListener("click", async () => {
	disable(elements)
	get.value = "Getting List..."

	try {
		list = await getList(user.value, "ANIME", token.value)
		list.entries = normalize(list.entries, list.min, list.max, 0, 1)
		get.value = getValue
	} catch {
		get.value = "Error!"
	}

	enable(elements)
})

train.addEventListener("click", async () => {
	disable(elements)
	train.value = "Training Model..."

	const callbacks = {
		onEpochEnd: (epoch) => {
			epoch = epoch / epochs.value * 100
			train.style.background = `linear-gradient(90deg, var(--acd), ${epoch}%, var(--acl), ${epoch}%, var(--acl))`
		},
		onTrainEnd: () => train.removeAttribute("style")
	}

	try {
		const database = await loadDatabase()
		const types = Array.from(type.options).map(option => option.value)
		const schema = createSchema(database, types)

		model = new Model(schema.length)
		data = prepareData(list.entries, database, schema)

		await model.train(data.train, epochs.value, patience.value, callbacks)
		train.value = trainValue
	} catch {
		train.value = "Error!"
	}

	enable(elements)
})

predict.addEventListener("click", async () => {
	disable(elements)
	predict.value = "Recommending..."
	main.replaceChildren()

	const dataPred = data.pred.filter(item =>
		item.type == type.value
		&& item.year >= yearFrom.value
		&& item.year <= yearTo.value
	)

	let preds, scores, scoresNorm

	try {
		preds = await model.predict(dataPred, limit.value)
		scores = preds.map(pred => pred.score)
		scoresNorm = normalize(scores, 0, 1, list.min, list.max)
		predict.value = predictValue
	} catch {
		predict.value = "Error!"
	}

	if (preds && scores && scoresNorm) {
		const header = document.createElement("output")
		const title = document.createElement("span")
		const score = document.createElement("span")

		title.innerText = "Title"
		score.innerText = "Score"

		header.appendChild(title)
		header.appendChild(score)
		main.appendChild(header)

		for (let i = 0; i < preds.length; i++) {
			const row = document.createElement("output")
			const link = document.createElement("a")
			const span = document.createElement("span")
			const score = document.createElement("span")

			link.href = preds[i].siteUrl
			link.target = "_blank"
			link.innerText = preds[i].title
			score.innerText = Math.round(scoresNorm[i])

			span.appendChild(link)
			row.appendChild(span)
			row.appendChild(score)
			main.appendChild(row)
		}
	}

	enable(elements)
})