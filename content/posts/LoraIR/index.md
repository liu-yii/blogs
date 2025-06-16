---
date : 2025-06-16T18:40:37+08:00
draft : false
title : 'LoraIR'
---
Daily Paper 002
第二天！！🥳

## Title: LoRA-IR: Taming Low-Rank Experts for Efficient All-in-One Image Restoration (ArXiv 2024)
## code
⭐⭐⭐⭐
## Abstract:
Prompt based all-in-one IR方法在处理真实场景中的复杂多变的退化时仍然存在挑战。文章提出了LoRA-IR，分为两部分：degradation-guided pretraining和parameter-efficient finetuning。

## Introduction
和之前的All-in-One方法类似，说明单一任务的模型很难在现实中不可预测和多变的环境中有效推广，并指出了之前的Prompt Learning IR的缺点：仅仅依靠轻量级的Prompt和static shared network很难捕捉到不同退化类型的细节和specific patterns. 并且不同退化类型之间的潜在相关性和共有的特征没有广泛利用. 

针对上面的问题，提出了LoRA-IR. 与其他方法的不同之处在于，LoRA-IR基于CLIP生成degradation prompt. 但是CLIP侧重于影像的全局语义信息，在应用于low-level vision时会出现性能不佳的情况. 为了解决这一问题，文章提出了DG-Router，将影像分为两个分支输入CLIP，下采样获取全局的信息，用sliding-window获取局部信息。 网络的训练分为两个阶段：1）用DG-Router得到的退化信息指导图像复原网络预训练；2）用LoRA对第一阶段得到的复原网络进行微调(基于MoE构建了一组low-rank restoration experts). 不同的专家模型增强了网络捕捉特定退化知识的能力，他们之间的协作则使网络具备了学习各种退化之间相关性的能力 (Different experts enhance the network’s ability to capture degradationspecific knowledge, while their collaboration equips the network with the capability to learn correlations between various degradations)。 

## Related Work
文章从IR model，VLM以及Parameter-efficient Fine-tuning (PEFT)三个方面介绍了相关工作。我这里就重点看了一下PEFT工作(对应LoRA)。


这样来看，LoRA原论文的这张图就非常形象地描述低秩投影的过程。

## Method
那其实文章的创新点在上面部分基本上都讲了，在方法部分主要是具体讲各部分的实现流程，就不详述了。具体方法可以参照原论文。

## Expirement Settings
这里文章对之前的All-in-One方法的实验设置进行了总结，分成5类：

4-task adverse weather removal; desnowing, deraining, dehazing and raindrop removal;
3-task real-word adverse weather removal: deraining, dehazing and desnowing;
3-task image restoration: deraining, dehazing and denoising;
5-task image restoration: deraining, low-light enhancement, desnowing, dehazing and deblurring;
10-task image restoration: deblurring dehazing, JPEG artifact removal, low-light enhancement, denoising, raindrop removal, deraining, shadow removal, desnowing, and inpainting.
这里放了部分实验结果，作者做了太多对比实验了，具体可以去看看原文。  

## Conclusion
我觉得文章整体创新性还挺高的，将CLIP全局-局部信息整合确实体现了simple but work.