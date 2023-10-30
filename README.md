# NExT-GPT: Any-to-Any Multimodal LLM

**Authors:** Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, Tat-Seng Chua  
**Affiliation:** NExT++, School of Computing, National University of Singapore  
**Project:** [NExT-GPT Project Page](https://next-gpt.github.io/)

---

## Intro:

Multimodal AI has become critical for human-like understanding of our world. However, most existing Multimodal Large Language Models only handle multimodal inputs, without flexible output abilities. In this work, NExT-GPT, an end-to-end any-to-any MM-LLM, is presented. It can accept and generate content in text, images, videos, and audio. This is achieved by connecting an LLM core with multimodal adaptors and diffusion decoders.

## Overview 
<p align="center">
  <img width="590" alt="Screenshot 2023-10-24 at 8 45 31 PM" src="https://github.com/minu803/NextGPT/assets/111295624/601582c4-08cf-4f32-a0cc-c0adf5442efa">
</p>


### Multimodal Input Encoding:
- Each input is processed via an encoder tailored for its modality, resulting in semantic embeddings of the input.
- Models used: ImageBind, which can process multiple modalities and yield semantic embeddings in the same embedding space. 

### LLM-centric Alignment:
- After obtaining the semantic embeddings, they are passed through small linear models called input projection models.
- The purpose of these models is to generate text from the embeddings that the LLM can understand.

### LLM-based Semantic Understanding:
- The core LLM produces the text response and also gives instructions for the generation of other modalities.
- LLM output may contain special tokens, [IMGi], [AUDi], and [VIDi] to indicate that output for a particular modality should be generated and to identify the portion of the LLM response related to that modality.
- Models used: Vicuna-7B, which is common among other multimodal LLMs. 

### Instruction-following Alignment:
- The LLM's output relevant to non-text modalities is processed by small transformer-based models called output projection models.
- These models convert the LLM outputs into representations suitable for the modality decoders.

### Multimodal Output Generation:
-  Outputs for each modality are generated using their specific diffusion decoder.
- Models used:
  - Image diffusion model: Stable Diffusion
  - Video model: Zeroscope
  - Audio generation model: AudioLDM


## Deeper Dive

### HOW?
**First**, we leverage established encoders to encode inputs in various modalities, where these representations are projected into language-like representations comprehensible to the LLM through a projection layer. 

**Second**, we harness an existing open-sourced LLM as the core to process input information for semantic understanding and reasoning. The LLM not only directly generates text tokens but also produces unique “modality signal” tokens that serve as instructions to dictate the decoding layers whether & what modal content to output correspondingly. 

**Third**, the produced multimodal signals with specific instructions, after projection, route to different encoders and finally generate content in corresponding modalities.

<p align="center">
  <img width="591" alt="Screenshot 2023-10-24 at 9 00 34 PM" src="https://github.com/minu803/NextGPT/assets/111295624/9fe9c7af-c2f1-4548-bf0d-cb0017b34b5b">
</p>

**QUESTION:** What is the advantage and disadvantage of using a trained model? Can you name the appropriate model to do the described task?

**ANSWER:** 🤔🤔


Only 1% of parameters corresponding to the input and output projections are updated during training with all the rest encoders and decoders frozen. This provides lightweight tuning for the overall framework.

Calculation: 131M(=4+33+31+31+32) / [131M + 12.275B(=1.2+7+1.3+1.8+0.975)], only 1% parameters are to be updated. This is also one of the key advantages of our MM-LLM.



### Lightweight Multimodal Alignment Learning:
<p align="center">
<img width="842" alt="Screenshot 2023-10-29 at 4 12 07 PM" src="https://github.com/minu803/NextGPT/assets/111295624/42a1f36e-27b2-4cdd-8f1f-1883fe5c96b1">
</p>

- Performs alignment between input/output projections and LLM core with minimal training
- **Encoding-side alignment:**
  - Input projection models using image-text, audio-text, and video-text pairs
  - Non-text input fed to the encoder yields a representation
  - Representation fed to input projection model to obtain aligned representation for LLM
  - LLM response compared to text caption, with loss propagated to the input projection model
- **Decoding-side alignment:**
  - Uses similar captioned inputs
  - LLM outputs a response with a signal token, processed through an output projection model (No generation of image, audio, or video)
  - Comparison made between output from projection model and encoding from text encoder of diffusion model (no actual diffusion process)
  - Minimizes distance between LLM signals and diffusion model text tokens


### Modality-Switching Instruction Tuning:
<p align="center">
<img width="883" alt="Screenshot 2023-10-29 at 4 13 19 PM" src="https://github.com/minu803/NextGPT/assets/111295624/a3849947-578e-4e1a-8c3b-266c462b0a9d">
</p>

- System fed with multi-modality dialogue inputs. Use (Input, OUTPUT) pairs
- LLM's response, with modality signal tokens, compared to gold annotations for LoRA weight updates
- Gold Annotations for LLM Outputs:
  - **Text+X to Text:** Input comprises text and another modality, with the output being caption text
  - **Text to Text+X:** Input is text, while the output is text combined with another modality
  - **High-Quality Dataset:** A dataset containing comprehensive multi-modality instructions and responses. This leverages GPT-4 for creating dialogues and incorporates images, videos, and audio where appropriate

## Result
<p align="center">
<img width="869" alt="Screenshot 2023-10-29 at 4 14 39 PM" src="https://github.com/minu803/NextGPT/assets/111295624/3a7fc12a-1985-4ca3-88d3-86ac450e62bb">
</p>

## Critical Analysis:
- Key strengths of NExT-GPT include end-to-end learning which avoids cascading errors of pipeline systems. The alignment learning minimizes the gap between modalities. Instruction tuning further empowers complex cross-modal capabilities.
- More competent in producing images, compared with the generations on videos and audio. Also generating mixed combinations of multimodal content is slightly inferior to the generation of single-modal content, due to the complexity.
- Limitations are that currently only 4 modalities are supported. This can be expanded to more like 3D, tables, etc. Also, the quality is limited by the diffusion model capabilities.
- Future work involves supporting more modalities and tasks and combining retrieval models to complement the generative process.

## Multimodal Pseudocode:
```
Algorithm: NextGPT(text, media, θ)

Input: 
  text: sequence of text prompt tokens 
  media: image/video/audio input
  
Hyperparameters:
  L: number of transformer layers
  H: number of attention heads
  de: token embedding dimension
  
Parameters θ:
  θLLaMa: pretrained LLaMa parameters (frozen) 
  θencoder: pretrained image/video/audio encoder parameters (frozen)
  θTextFc: trainable fully-connected alignment modules
  
Output:
  continuation: generated continuation text
  generation: generated image/video/audio
  
# Stage 1: Caption training
for iter in 1..N1:
  for (text, media) in dataset:
    
    encoding = Encoder(media) # Encode media
    input = [text, '[IMG]', encoding] # Concatenate
     
    p = Transformer(input; θLLaMa) # Forward pass
    loss = caption_loss(p, text) # Language modeling loss
    
    update(θLLaMa, ∇loss) # Update only LLaMa
  
# Stage 2: Alignment training  
for iter in 1..N2:
  for media in dataset:
  
    encoding = Encoder(media)
    fake_text = '[IMG]'
    img_embedding = Transformer(fake_text; θLLaMa)
    loss = ||img_embedding - encoding||22 
    update(θTextFc, ∇loss) # Update alignment modules

# Stage 3: Joint training
for iter in 1..N3:
  for (text, media) in dataset:
  
    text_embedding = Transformer(text; θLLaMa) 
    img_embedding = Transformer('[IMG]'; θLLaMa)
    img_encoding = Encoder(media)
    img_embedding = TextFc(img_embedding) # Align
      
    encoding = [text_embedding, img_embedding] 
    p = Transformer(encoding; θLLaMa)
    loss = caption_loss(p, text) 
    update(θLLaMa, θTextFc, ∇loss)
    
# Inference
prompt = "[Text prompt]"
text_embedding = Transformer(prompt; θLLaMa)

img_embedding = Transformer('[IMG]'; θLLaMa) 
img_embedding = TextFc(img_embedding)
generation = SD(img_embedding) # Generate image

encoding = [text_embedding, img_embedding]
continuation = sample(Transformer(encoding; θLLaMa))
```

- Leverage pre-trained models - LLaMa for text, ImageBind encoder for images.
- Align modalities using trainable modules - TextFcLayers.
- Jointly train on a multi-modal task like instruction following.
- At inference, generate text and images by extracting embeddings at special tokens.

## Code Demonstration
You can find the necessary code within `anyToImageVideoAudio.py`

## Video Demonstration
[![YouTube Video](https://img.youtube.com/vi/aqw2SCWeWD0/maxresdefault.jpg)](https://www.youtube.com/watch?v=aqw2SCWeWD0)


## Citations & Resrouces

- AI Papers Academy. (2023, September 16). Next-GPT: Any-to-any multimodal LLM. YouTube. https://www.youtube.com/watch?v=UtN1hOMces&t=199s&ab_channel=AIPapersAcademy 
- Akkus, C., Chu, L., Djakovic, V., Jauch-Walser, S., Koch, P., Loss, G., Marquardt, C., Moldovan, M., Sauter, N., Schneider, M., Schulte, R., Urbanczyk, K., Goschenhofer, J., Heumann, C., Hvingelby, R., Schalk, D., & Aßenmacher, M. (2023). Multimodal deep learning. arXiv preprint arXiv:2301.04856. https://arxiv.org/abs/2301.04856
- van der Maaten, L., & Misra, I. (n.d.). Advances in multimodal understanding research at meta ai. AI at Meta. https://ai.meta.com/blog/advances-in-multimodal-understanding-research-at-meta-ai/ 
- Wu, S., Fei, H., Qu, L., Ji, W., & Chua, T.-S. (2023, September 13). Next-GPT: Any-to-any multimodal LLM. arXiv.org. https://arxiv.org/abs/2309.05519 
- Xu, P., Zhu, X., & Clifton, D. A. (2023, May 10). Multimodal learning with transformers: A survey. arXiv.org. https://arxiv.org/abs/2206.06488 




