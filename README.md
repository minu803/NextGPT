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

**QUESTION:** What is the advantage and disadvantage of using a trained model? Can you name the appropriate model to do the described task?

**ANSWER:** We avoid cold-start training and facilitate the potential growth of more modalities.

### Multimodal Input Encoding:
- Existing encoders like ImageBind are used to encode inputs in various modalities (text, image, video, audio)
- Input projection layers map the encodings into a common language space understandable to the LLM core

### LLM-centric Alignment:
- The input projections are trained via image/video/audio captioning tasks
- This aligns the input representations with the textual feature space of the LLM

### LLM-based Semantic Understanding:
- The aligned multimodal inputs are fed to the LLM core (e.g. ViCuNA)
- The LLM performs high-level reasoning and understanding over the multimodal contexts

### Instruction-following Alignment:
- The output projections are trained by minimizing the distance between LLM signal tokens and conditioned text tokens of diffusion models
- This enables diffusion models to accurately interpret LLM's instructions

### Multimodal Output Generation:
- Based on signals from LLM, diffusion models like SD, AudioLDM, and Zeroscope generate outputs
- Different decoders used for image, video, and audio synthesis
- Overall end-to-end training enhances coherent cross-modal content creation

## Deeper Dive

### HOW?
First, we leverage established encoders to encode inputs in various modalities, where these representations are projected into language-like representations comprehensible to the LLM through a projection layer. 

Second, we harness an existing open-sourced LLM as the core to process input information for semantic understanding and reasoning. The LLM not only directly generates text tokens but also produces unique “modality signal” tokens that serve as instructions to dictate the decoding layers whether & what modal content to output correspondingly. 

Third, the produced multimodal signals with specific instructions, after projection, route to different encoders and finally generate content in corresponding modalities.

<p align="center">
  <img width="591" alt="Screenshot 2023-10-24 at 9 00 34 PM" src="https://github.com/minu803/NextGPT/assets/111295624/9fe9c7af-c2f1-4548-bf0d-cb0017b34b5b">
</p>



### Lightweight Multimodal Alignment Learning:
- Performs alignment between input/output projections and LLM core with minimal training
- Encoding-side alignment: Trains input projections via image/video/audio captioning
- Aligns multimodal encoder representations with LLM's textual feature space
- Decoding-side alignment: Minimizes distance between LLM signals and diffusion model text tokens
- Enables diffusion models to interpret LLM instructions accurately
- Only input and output projection layers trained - efficient lightweight tuning

### Modality-Switching Instruction Tuning:
- Introduces novel Modality-Switching Instruction Tuning (MosIT)
- Manual high-quality dataset with diverse cross-modal instructions
- Empowers model with complex human-like multimodal reasoning abilities
- Tuning involves end-to-end training on MosIT data
- Updates both projections and LLM parameters
- Result: Improved cross-modal understanding and generation from instructions

## Critical Analysis:
- Key strengths of NExT-GPT include end-to-end learning which avoids cascading errors of pipeline systems. The alignment learning minimizes the gap between modalities. Instruction tuning further empowers complex cross-modal capabilities.
- Limitations are that currently only 4 modalities are supported. This can be expanded to more like 3D, tables, etc. Also, the quality is limited by the diffusion model capabilities.
- Future work involves supporting more modalities and tasks and combining retrieval models to complement the generative process.

## Multimodal Psudocode:
```
Algorithm: NextGPT
Input: 
- x ∈ V∗, a sequence of token IDs

Output: 
- P ∈ [0,1]^(NV x length(x)): P:t represents p̂θ(x(t+1) | x(1:t))

Hyperparameters:
- θmax
- L: Number of layers
- H: Number of heads (assuming multi-head attention, but this isn't explicitly mentioned)
- de: Dimension of embeddings
- dmlp: Dimension of MLP

Parameters θ:
1. We ∈ R^(de x NV): Token embeddings
2. Wp ∈ R^(de x θmax): Positional embeddings
3. For each layer l from 1 to L:
   - Wl: Multi-head self-attention weights
   - γl1, βl1, γl2, βl2 ∈ R^de: Layer normalization parameters
   - Wlmlp1 ∈ R^(dmlp x de), blmlp1 ∈ R^dmlp: MLP weights
   - Wlmlp2 ∈ R^(de x dmlp), blmlp2 ∈ R^de: MLP weights
4. γ, β ∈ R^de: Final layer normalization parameters
5. Wu ∈ R^(NV x de): Unembedding matrix

Procedure:
1. Set θ to length(x)
2. For each token t from 1 to θ:
   - Compute embeddings: et = We:x(t) + Wp:t
3. Concatenate embeddings: X = [e1, ..., eθ]
4. For each layer l from 1 to L:
   4.1. For each token t from 1 to θ:
        - Normalize: X̃:t = layer_norm(X:t | γl1, βl1)
        - Update X with self-attention: X += MHAttention(X̃ | Wl, Mask)
   4.2. For each token t from 1 to θ:
        - Normalize: X̃:t = layer_norm(X:t | γl2, βl2)
        - MLP update: X += (Wlmlp2 · GELU(Wlmlp1 · X̃ + blmlp1)^T + blmlp2)^T
5. For each token t from 1 to θ:
   - Apply final layer normalization: X:t = layer_norm(X:t | γ, β)
6. Compute output: P = softmax(Wu · X)

Return P
```

## Code Demonstration

## Video Demonstration

## Resrouces


