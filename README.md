# GigaChat Plugin for Dify

## Overview

GigaChat is a Russian AI assistant developed by Sber. This plugin provides integration with GigaChat models for text
generation and embeddings.

## Configuration

After installing the GigaChat plugin, you need to configure it with your credentials:

1. Get your API key from the [GigaChat personal cabinet](https://developers.sber.ru/portal/products/gigachat)
2. Enter the credentials in the Model Provider settings
3. Choose the appropriate scope for your account type

### Parameters

- **API Key**: Your GigaChat authorization key
- **Scope**: API version scope
    - `GIGACHAT_API_PERS` - for personal accounts
    - `GIGACHAT_API_B2B` - for business accounts (prepaid)
    - `GIGACHAT_API_CORP` - for corporate accounts (postpaid)
- **Base URL** (optional): Custom API endpoint (default: https://gigachat.devices.sberbank.ru/api/v1)
- **Verify SSL**: Whether to verify SSL certificates (default: false)

## Supported Models

### Language Models (Second Generation)

All language models support 128K tokens context size:

- **GigaChat-2** - Fast and lightweight model for simple everyday tasks
- **GigaChat-2-Pro** - Advanced model for resource-intensive tasks with enhanced data processing, creativity, and
  instruction following
- **GigaChat-2-Max** - Powerful model for the most complex and large-scale tasks requiring highest creativity and
  quality

### Embedding Models

- **Embeddings** - Basic embedding model (512 tokens context)
- **EmbeddingsGigaR** - Advanced embedding model with larger context (4096 tokens)

## Features

- Text generation with streaming support
- Multi-turn chat conversations
- Function calling (tool use)
- Vision capabilities (image analysis)
- Text embeddings
- Custom SSL certificate handling
- Early access models support (with `-preview` suffix)

## Model Selection Guide

Choose the appropriate model based on your needs:

- **GigaChat-2**: Best for simple tasks requiring maximum speed and lower cost
- **GigaChat-2-Pro**: Ideal for complex instructions, summarization, text editing, and comprehensive Q&A
- **GigaChat-2-Max**: For advanced tasks requiring high creativity and quality

## Notes

- The first-generation models (GigaChat, GigaChat-Plus, GigaChat-Pro, GigaChat-Max) are deprecated and automatically
  redirect to their second-generation equivalents
- Pricing remains the same as the first-generation models
- Models are regularly updated with new capabilities through early access versions