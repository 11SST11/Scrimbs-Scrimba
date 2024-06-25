import { HfInference } from '@huggingface/inference'

// Create your Hugging Face Token: https://huggingface.co/settings/tokens
// Set your Hugging Face Token: https://scrimba.com/dashboard#env
// Learn more: https://scrimba.com/links/env-variables


// Hugging Face Inference API docs: https://huggingface.co/docs/huggingface.js/inference/README
const HF_TOKEN="XxX"
const hf = new HfInference(process.env.HF_TOKEN)

// console.log(hf)

const textToGenerate = "The definition of machine learning inference is "
// an asynchronous call 
const response = await hf.textGeneration({
    inputs: textToGenerate,
    // model:"microsoft/Florence-2-large"
})
console.log(response)


const textToTranslate = "It's an exciting time to be an AI engineer"

// https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages-covered

const textTranslationResponse = await hf.translation({
  model: 'facebook/mbart-large-50-many-to-many-mmt',
  inputs: textToTranslate,
  parameters: {
      src_lang: "en_XX",
      tgt_lang: "es_XX"
  }
})
const textToClassify = "I just bought a new camera. It's the best camera I've ever owned!"

const response = await hf.textClassification({
    model: "distilbert-base-uncased-finetuned-sst-2-english",
    inputs: textToClassify
})