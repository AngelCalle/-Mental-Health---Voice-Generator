import { pipeline } from '@xenova/transformers';
import wavefile from 'wavefile';
import fs from 'fs';
import fetch from 'node-fetch';

async function getSpeakerEmbeddings(url) {
	const response = await fetch(url);
	const buffer = await response.arrayBuffer();

	// Asumiendo que cada embedding es un Float32, necesitamos 512 * 4 bytes
	const requiredLength = 512 * 4;

	let embeddingsBuffer = buffer;
	if (buffer.byteLength < requiredLength) {
		// Si el buffer es demasiado pequeño, lo extendemos y rellenamos con ceros
		embeddingsBuffer = new ArrayBuffer(requiredLength);
		new Uint8Array(embeddingsBuffer).set(new Uint8Array(buffer));
	} else if (buffer.byteLength > requiredLength) {
		// Si el buffer es demasiado grande, lo recortamos
		embeddingsBuffer = buffer.slice(0, requiredLength);
	}

	return new Float32Array(embeddingsBuffer);
}

async function generateSpeech() {
	try {
		const EMBED_URL =
			'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/KGSAGAR/speecht5_finetuned_voxpopuli_es.bin';
		const PHRASE =
			'Recuerda, esta solución asume ciertos detalles sobre los embeddings y el modelo que podrían no ser precisos. Si este enfoque no funciona, te recomendaría consultar la documentación específica del modelo o buscar ejemplos de código que utilicen este modelo con embeddings personalizados para obtener más orientación';
		const synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', { quantized: false });
		const speakerEmbeddings = await getSpeakerEmbeddings(EMBED_URL);

		const output = await synthesizer(PHRASE, { speaker_embeddings: speakerEmbeddings });

		const wav = new wavefile.WaveFile();
		wav.fromScratch(1, output.sampling_rate, '32f', output.audio);
		fs.writeFileSync('output.wav', wav.toBuffer());
	} catch (error) {
		console.error('Error al generar el discurso:', error);
	}
}

generateSpeech();
