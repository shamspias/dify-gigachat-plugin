# GigaChat Plugin for Dify

## Overview

GigaChat is a Russian AI assistant developed by Sber. This plugin integrates GigaChat’s text‑generation and embedding models into the Dify platform, allowing you to use GigaChat as a model provider in your own Dify applications.

* Easy integration into Dify’s LLM and embedding pipeline
* Compatible with GigaChat API v1
* Supports text generation, chat, vision, function calling, and embeddings
* Supports both first and second-generation models (with deprecation notices)

## Configuration

After installing the GigaChat plugin, configure it with your credentials:

1. Get your API key from the [GigaChat personal cabinet](https://developers.sber.ru/portal/products/gigachat)
2. Enter the credentials in the Model Provider settings
3. Choose the appropriate scope for your account type

### Parameters

* **API Key**: Your GigaChat authorization key
* **Scope**: API version scope

  * `GIGACHAT_API_PERS` - for personal accounts
  * `GIGACHAT_API_B2B` - for business accounts (prepaid)
  * `GIGACHAT_API_CORP` - for corporate accounts (postpaid)
* **Base URL** (optional): Custom API endpoint (default: [https://gigachat.devices.sberbank.ru/api/v1](https://gigachat.devices.sberbank.ru/api/v1))
* **Verify SSL**: Whether to verify SSL certificates (default: true)

## Supported Models

### Language Models (Second Generation)

*All language models support 128K tokens context size:*

* **GigaChat-2** – Fast and lightweight for simple everyday tasks
* **GigaChat-2-Pro** – Advanced, resource-intensive, for creativity and instruction following
* **GigaChat-2-Max** – For the most complex, large-scale tasks requiring highest quality

### Embedding Models

* **Embeddings** – Basic embedding model (512 tokens context)
* **EmbeddingsGigaR** – Advanced embedding model with 4096 token context

### Deprecated (First Generation)

The first-generation models (GigaChat, GigaChat-Plus, GigaChat-Pro, GigaChat-Max) are deprecated and redirect to their second-generation equivalents.

## Features

* Text generation with streaming support
* Multi-turn chat conversations
* Function calling (tool use)
* Vision capabilities (image analysis)
* Text embeddings
* Custom SSL certificate handling
* Early access/preview model support (with `-preview` suffix)

## Model Selection Guide

Choose the appropriate model based on your needs:

| Model          | Best For                                          |
| -------------- | ------------------------------------------------- |
| GigaChat-2     | Simple tasks, max speed, lower cost               |
| GigaChat-2-Pro | Complex instructions, summarization, text editing |
| GigaChat-2-Max | Highest creativity, quality, advanced reasoning   |

## Notes

* Pricing remains the same as the first-generation models
* Models are regularly updated with new capabilities (including early access)
* Always check the [Sber developer portal](https://developers.sber.ru/) for the latest model support and deprecation info

## Privacy & Data Handling

* The plugin only uses your provided API key and configuration to connect to GigaChat.
* User prompts and model outputs are transmitted to Sber’s API for processing but are **not** stored or logged by the plugin.
* See [PRIVACY.md](./PRIVACY.md) for full policy and Sber’s [GigaChat Service Terms](https://developers.sber.ru/docs/ru/policies/privacy-policy) and [Data Policy Gigachat](https://www.sberbank.ru/privacy/policy#pdn).

## Contribution & License

Contributions are welcome! Please open issues or PRs. See `LICENSE` for license details.

## Contact

* [Sber Developer Portal](https://developers.sber.ru/)
* For plugin bugs: open a GitHub Issue on this repo.

---

*This plugin is an open source project, not officially endorsed by Sber. All trademarks are property of their respective owners.*
