# GigaChat Plugin for Dify

## Overview

GigaChat is a Russian AI assistant developed by Sber. This plugin provides integration with GigaChat models for text generation and embeddings.

## Configuration

After installing the GigaChat plugin, you need to configure it with your credentials:

1. Get your API key from the GigaChat personal cabinet
2. Enter the credentials in the Model Provider settings
3. Choose the appropriate scope for your account type

### Parameters

- **API Key**: Your GigaChat authorization key
- **Scope**: API version scope
  - `GIGACHAT_API_PERS` - for personal accounts
  - `GIGACHAT_API_B2B` - for business accounts (prepaid)
  - `GIGACHAT_API_CORP` - for corporate accounts (postpaid)
- **Base URL** (optional): Custom API endpoint
- **Verify SSL**: Whether to verify SSL certificates

## Supported Models

### Language Models
- GigaChat - Basic model
- GigaChat-Plus - Enhanced model
- GigaChat-Pro - Professional model
- GigaChat-Max - Maximum capability model

### Embedding Models
- GigaChat-Embeddings - Text embedding model

## Features

- Text generation
- Chat conversations
- Function calling
- Streaming responses
- Text embeddings
- Image analysis
- Custom SSL certificate handling