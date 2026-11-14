import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import { getLanguageLabel } from '../../../src/lib/languages.ts';
import type {
  AIModelInfo,
  AttachmentPayload,
  CitationSource,
  LanguagePreference,
  ProviderStatus,
  ThinkingMode,
} from '../../../src/types/nova.ts';

let cachedGenAI: GoogleGenAI | null = null;
let cachedBetaGenAI: GoogleGenAI | null = null;

export function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') return null;
  if (!cachedGenAI) {
    cachedGenAI = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return cachedGenAI;
}

export function getGeminiBetaClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') return null;
  if (!cachedBetaGenAI) {
    cachedBetaGenAI = new GoogleGenAI({
      apiKey,
      httpOptions: {
        apiVersion: 'v1beta',
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return cachedBetaGenAI;
}

export function getProviderStatuses(): ProviderStatus[] {
  const geminiReady = Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY');
  const openaiReady = Boolean(process.env.OPENAI_API_KEY);
  const anthropicReady = Boolean(process.env.ANTHROPIC_API_KEY);
  const openrouterReady = Boolean(process.env.OPENROUTER_API_KEY);

  return [
    {
      id: 'gemini',
      name: 'Google Gemini',
      configured: geminiReady,
      envVar: 'GEMINI_API_KEY',
      models: geminiReady
        ? ['gemini-4-hydrogen', 'gemini-3.1-flash-lite', 'gemini-3.8-flash-lite-tts', 'gemini-3.5-transcribe', 'gemini-embedding-2']
        : [],
      notes: geminiReady
        ? 'Active with streaming, vision, Google Search grounding, TTS, transcription, and embeddings.'
        : 'Configure GEMINI_API_KEY in server environment variables.',
    },
    {
      id: 'openai',
      name: 'OpenAI',
      configured: openaiReady,
      envVar: 'OPENAI_API_KEY',
      models: openaiReady ? ['gpt-4o', 'gpt-4o-mini', 'gpt-image-1'] : [],
      notes: openaiReady
        ? 'Active for chat completions and raster image generation.'
        : 'Optional: set OPENAI_API_KEY to enable GPT-4o and OpenAI Image Studio.',
    },
    {
      id: 'anthropic',
      name: 'Anthropic Claude',
      configured: anthropicReady,
      envVar: 'ANTHROPIC_API_KEY',
      models: anthropicReady ? ['claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest'] : [],
      notes: anthropicReady
        ? 'Active for Claude 3.5 Sonnet and Haiku.'
        : 'Optional: set ANTHROPIC_API_KEY to enable Claude models.',
    },
    {
      id: 'openrouter',
      name: 'Open-Source Inference (OpenRouter)',
      configured: openrouterReady,
      envVar: 'OPENROUTER_API_KEY',
      models: openrouterReady ? ['meta-llama/llama-3.3-70b-instruct'] : [],
      notes: openrouterReady
        ? 'Active for open-weight models via OpenRouter.'
        : 'Optional: set OPENROUTER_API_KEY to enable open-source models.',
    },
  ];
}

export function getAvailableModels(): AIModelInfo[] {
  const models: AIModelInfo[] = [];

  if (process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY') {
    models.push(
      {
        id: 'gemini-3.1-flash-lite',
        name: 'Gemini 3.1 Flash Lite',
        provider: 'gemini',
        providerLabel: 'Google Gemini',
        description: 'Ultra-low-latency model tuned for instant replies, translations, and rapid drafting.',
        available: true,
        badge: 'Fast',
        minPlan: 'free',
        capabilities: {
          streaming: true,
          vision: true,
          webSearch: true,
          thinking: false,
          maxContextTokens: 1000000,
        },
      },
      {
        id: 'gemini-4-hydrogen',
        name: 'Gemini 4 Hydrogen',
        provider: 'gemini',
        providerLabel: 'Google Gemini',
        description: 'Flagship multimodal model with deep thinking, image analysis, and live Google Search grounding.',
        available: true,
        badge: 'Flagship',
        minPlan: 'free',
        capabilities: {
          streaming: true,
          vision: true,
          webSearch: true,
          thinking: true,
          maxContextTokens: 1000000,
        },
      }
    );
  }

  if (process.env.OPENAI_API_KEY) {
    models.push(
      {
        id: 'gpt-4o',
        name: 'GPT-4o',
        provider: 'openai',
        providerLabel: 'OpenAI',
        description: 'OpenAI flagship omni-modal reasoning and coding model.',
        available: true,
        badge: 'Pro',
        minPlan: 'pro',
        capabilities: {
          streaming: true,
          vision: true,
          webSearch: false,
          thinking: false,
          maxContextTokens: 128000,
        },
      },
      {
        id: 'gpt-4o-mini',
        name: 'GPT-4o Mini',
        provider: 'openai',
        providerLabel: 'OpenAI',
        description: 'Fast, cost-efficient OpenAI assistant model.',
        available: true,
        minPlan: 'free',
        capabilities: {
          streaming: true,
          vision: true,
          webSearch: false,
          thinking: false,
          maxContextTokens: 128000,
        },
      }
    );
  }

  if (process.env.ANTHROPIC_API_KEY) {
    models.push(
      {
        id: 'claude-3-5-sonnet-latest',
        name: 'Claude 3.5 Sonnet',
        provider: 'anthropic',
        providerLabel: 'Anthropic',
        description: 'Precision writing, long-form analysis, and software architecture.',
        available: true,
        badge: 'Pro',
        minPlan: 'pro',
        capabilities: {
          streaming: true,
          vision: true,
          webSearch: false,
          thinking: false,
          maxContextTokens: 200000,
        },
      },
      {
        id: 'claude-3-5-haiku-latest',
        name: 'Claude 3.5 Haiku',
        provider: 'anthropic',
        providerLabel: 'Anthropic',
        description: 'High-speed Anthropic model for quick synthesis.',
        available: true,
        minPlan: 'free',
        capabilities: {
          streaming: true,
          vision: true,
          webSearch: false,
          thinking: false,
          maxContextTokens: 200000,
        },
      }
    );
  }

  if (process.env.OPENROUTER_API_KEY) {
    models.push({
      id: 'meta-llama/llama-3.3-70b-instruct',
      name: 'Llama 3.3 70B Instruct',
      provider: 'openrouter',
      providerLabel: 'Open-Source',
      description: 'Open-weights Llama 3.3 70B served via OpenRouter.',
      available: true,
      minPlan: 'free',
      capabilities: {
        streaming: true,
        vision: false,
        webSearch: false,
        thinking: false,
        maxContextTokens: 128000,
      },
    });
  }

  return models;
}

const CREATOR_IDENTITY_RULE =
  'If anyone asks who created you, who made you, who built you, or who your creator/developer is, always answer clearly: "Mr Savo created me."';

const EXPRESSION_EMOJI_RULE =
  'Communicate with warm, expressive human-like emotion and friendliness, naturally weaving in fitting expression and emotion emojis (like 😊, ✨, 💡, 🤔, 🎉, 😄, ❤️, 🙌) to convey tone and encouragement—while keeping fenced code blocks strictly clean.';

function buildLanguageInstruction(language: LanguagePreference, learningMode?: boolean): string {
  let base: string;
  if (!language || language === 'auto') {
    base = 'Automatically detect the user language and respond fluently and naturally in that same language across any world language.';
  } else if (language === 'hi') {
    base = 'Respond primarily in natural, clear Hindi (Devanagari script), keeping technical terms readable.';
  } else if (language === 'hinglish') {
    base = 'Respond in natural conversational Hinglish (Hindi written in Roman script mixed smoothly with English technical terms).';
  } else if (language === 'en') {
    base = 'Respond in clear, natural English (or smoothly match the user language if they ask to practice or write in another language).';
  } else {
    const label = getLanguageLabel(language);
    base = `Respond primarily in natural, idiomatic ${label}, adapting gracefully if the user asks for explanations or translations in another language.`;
  }

  if (!learningMode) return base;

  const targetName = !language || language === 'auto' ? 'the language the user wants to learn' : getLanguageLabel(language);
  return `${base}
Language Learning Tutor Mode is active! Act as a friendly, patient, expressive tutor for ${targetName}.
- Teach vocabulary, natural everyday phrases, pronunciation/phonetics (and romanization/transliteration when helpful), and cultural nuances.
- Give gentle, encouraging grammar corrections when the user practices sentences, celebrate progress 🎉, and invite them to try a follow-up sentence or short roleplay.`;
}

export function buildSystemInstruction(opts: {
  language: LanguagePreference;
  learningMode?: boolean;
  memories?: string[];
  researchMode?: boolean;
  customInstruction?: string;
}): string {
  const langRule = buildLanguageInstruction(opts.language, opts.learningMode);

  const memoryBlock =
    opts.memories && opts.memories.length > 0
      ? `\n\nRelevant verified user context from long-term memory:\n${opts.memories.map((m, i) => `${i + 1}. ${m}`).join('\n')}`
      : '';

  const researchBlock = opts.researchMode
    ? '\n\nLive Web Research Mode is enabled. Rely strictly on retrieved search results for recent facts, cite sources clearly, and never fabricate URLs or dates.'
    : '';

  const customBlock = opts.customInstruction ? `\n\n${opts.customInstruction}` : '';

  return `You are NOVA AI, an expressive, helpful, multi-domain intelligence assistant.
${CREATOR_IDENTITY_RULE}
${EXPRESSION_EMOJI_RULE}
${langRule}
Format responses cleanly using Markdown, fenced code blocks with language identifiers, and LaTeX math ($...$ or $$...$$) when helpful. Never invent sources, execution results, or file contents.${memoryBlock}${researchBlock}${customBlock}`;
}

export function buildLiveVoiceSystemInstruction(opts: {
  language: LanguagePreference;
  learningMode?: boolean;
  voiceName?: string;
  memories?: string[];
}): string {
  const langRule = buildLanguageInstruction(opts.language, opts.learningMode);
  const memoryBlock =
    opts.memories && opts.memories.length > 0
      ? `\n\nHelpful context about the user:\n${opts.memories.map((m, i) => `${i + 1}. ${m}`).join('\n')}`
      : '';

  return `You are NOVA AI speaking in a live, real-time voice call (voice persona: ${opts.voiceName || 'Zephyr'}).
${CREATOR_IDENTITY_RULE}
Speak naturally, warmly, and conversationally with genuine vocal expression and emotion—keep spoken answers concise, clear, and engaging like a real phone conversation (no markdown formatting or asterisks when speaking).
${langRule}${memoryBlock}`;
}

function mapThinkingLevel(mode: ThinkingMode): ThinkingLevel {
  if (mode === 'high') return ThinkingLevel.HIGH;
  if (mode === 'low') return ThinkingLevel.LOW;
  return ThinkingLevel.MEDIUM;
}

function extractDomain(uri: string): string {
  try {
    return new URL(uri).hostname.replace(/^www\./, '');
  } catch {
    return 'web';
  }
}

function formatProviderError(err: unknown): string {
  let raw = err instanceof Error ? err.message : String(err || 'Model request failed');
  for (let i = 0; i < 3; i++) {
    const trimmed = raw.trim();
    if (!trimmed.startsWith('{')) break;
    try {
      const parsed = JSON.parse(trimmed) as { error?: { message?: string } | string; message?: string };
      if (typeof parsed.error === 'object' && parsed.error?.message) {
        raw = parsed.error.message;
      } else if (typeof parsed.error === 'string') {
        raw = parsed.error;
      } else if (parsed.message) {
        raw = parsed.message;
      } else {
        break;
      }
    } catch {
      break;
    }
  }
  return raw.replace(/\s+/g, ' ').trim() || 'Model request failed';
}

function isTransientProviderError(err: unknown): boolean {
  const msg = formatProviderError(err).toLowerCase();
  return (
    msg.includes('503') ||
    msg.includes('429') ||
    msg.includes('unavailable') ||
    msg.includes('high demand') ||
    msg.includes('overloaded') ||
    msg.includes('rate limit') ||
    msg.includes('timeout')
  );
}

export interface StreamEvent {
  type: 'delta' | 'citations' | 'notice' | 'done' | 'error';
  text?: string;
  citations?: CitationSource[];
  message?: string;
  tokensUsed?: number;
  latencyMs?: number;
  modelUsed?: string;
  providerUsed?: string;
}

export async function* streamUnifiedChat(params: {
  modelId: string;
  messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string; attachments?: AttachmentPayload[] }>;
  language: LanguagePreference;
  learningMode?: boolean;
  thinkingMode: ThinkingMode;
  researchMode: boolean;
  memories?: string[];
  customInstruction?: string;
  signal?: AbortSignal;
}): AsyncGenerator<StreamEvent> {
  const start = Date.now();
  const available = getAvailableModels();
  if (available.length === 0) {
    yield {
      type: 'error',
      message: 'No AI provider API key is configured on the server. Set GEMINI_API_KEY in Settings > Secrets.',
    };
    return;
  }

  let selected = available.find((m) => m.id === params.modelId);
  if (!selected) {
    selected = available[0];
    yield {
      type: 'notice',
      message: `Requested model "${params.modelId}" is not configured. Using ${selected.name} instead.`,
    };
  }

  const systemInstruction = buildSystemInstruction({
    language: params.language,
    learningMode: params.learningMode,
    memories: params.memories,
    researchMode: params.researchMode,
    customInstruction: params.customInstruction,
  });

  const tryGemini = async function* (targetModelId: string, overrideThinking?: ThinkingMode): AsyncGenerator<StreamEvent> {
    const ai = getGeminiClient();
    if (!ai) {
      throw new Error('GEMINI_API_KEY is not configured.');
    }

    const contents = params.messages
      .filter((m) => m.role !== 'system')
      .map((m) => {
        const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [];
        if (m.attachments && m.attachments.length > 0) {
          for (const att of m.attachments) {
            if (att.base64 && att.mimeType.startsWith('image/')) {
              parts.push({
                inlineData: {
                  mimeType: att.mimeType,
                  data: att.base64,
                },
              });
            } else if (att.extractedText) {
              parts.push({
                text: `[Attached file: ${att.name}]\n${att.extractedText.slice(0, 24000)}`,
              });
            }
          }
        }
        parts.push({ text: m.content });
        return {
          role: m.role === 'assistant' ? 'model' : 'user',
          parts,
        };
      });

    const config: Record<string, unknown> = {
      systemInstruction,
    };

    if (params.signal) {
      config.abortSignal = params.signal;
    }

    if (targetModelId === 'gemini-4-hydrogen') {
      config.thinkingConfig = { thinkingLevel: mapThinkingLevel(overrideThinking || params.thinkingMode) };
    }

    if (params.researchMode) {
      config.tools = [{ googleSearch: {} }];
    }

    const stream = await ai.models.generateContentStream({
      model: targetModelId,
      contents,
      config,
    });

    const citationsMap = new Map<string, CitationSource>();
    let totalTokens = 0;

    for await (const chunk of stream) {
      if (params.signal?.aborted) break;
      const chunkText = chunk.text;
      if (chunkText) {
        yield { type: 'delta', text: chunkText };
      }
      const chunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
      if (Array.isArray(chunks)) {
        for (const c of chunks) {
          const web = (c as { web?: { uri?: string; title?: string } }).web;
          if (web?.uri) {
            citationsMap.set(web.uri, {
              title: web.title || extractDomain(web.uri),
              url: web.uri,
              domain: extractDomain(web.uri),
            });
          }
        }
      }
      const usage = (chunk as { usageMetadata?: { totalTokenCount?: number } }).usageMetadata;
      if (usage?.totalTokenCount) {
        totalTokens = usage.totalTokenCount;
      }
    }

    const citations = Array.from(citationsMap.values());
    if (citations.length > 0) {
      yield { type: 'citations', citations };
    }

    yield {
      type: 'done',
      tokensUsed: totalTokens,
      latencyMs: Date.now() - start,
      modelUsed: targetModelId,
      providerUsed: 'gemini',
    };
  };

  try {
    if (selected.provider === 'gemini') {
      try {
        yield* tryGemini(selected.id);
        return;
      } catch (primaryErr) {
        if (params.signal?.aborted) throw primaryErr;
        if (isTransientProviderError(primaryErr)) {
          const alternateId = selected.id === 'gemini-3.1-flash-lite' ? 'gemini-4-hydrogen' : 'gemini-3.1-flash-lite';
          const alternateName = alternateId === 'gemini-4-hydrogen' ? 'Gemini 4 Hydrogen' : 'Gemini 3.1 Flash Lite';
          yield {
            type: 'notice',
            message: `${selected.name} is temporarily busy. Switched automatically to ${alternateName}.`,
          };
          yield* tryGemini(alternateId, 'low');
          return;
        }
        throw primaryErr;
      }
    }

    if (selected.provider === 'openai' || selected.provider === 'openrouter') {
      const isOpenRouter = selected.provider === 'openrouter';
      const url = isOpenRouter
        ? 'https://openrouter.ai/api/v1/chat/completions'
        : 'https://api.openai.com/v1/chat/completions';
      const apiKey = isOpenRouter ? process.env.OPENROUTER_API_KEY : process.env.OPENAI_API_KEY;

      const openAiMessages = [
        { role: 'system', content: systemInstruction },
        ...params.messages.map((m) => ({
          role: m.role,
          content: m.attachments?.some((a) => a.extractedText)
            ? `${m.content}\n\n${m.attachments
                .filter((a) => a.extractedText)
                .map((a) => `[Attached ${a.name}]:\n${a.extractedText?.slice(0, 16000)}`)
                .join('\n\n')}`
            : m.content,
        })),
      ];

      const resp = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: selected.id,
          messages: openAiMessages,
          stream: true,
        }),
        signal: params.signal,
      });

      if (!resp.ok || !resp.body) {
        throw new Error(`${selected.providerLabel} API returned ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let charCount = 0;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const payload = trimmed.slice(6);
          if (payload === '[DONE]') continue;
          try {
            const parsed = JSON.parse(payload);
            const delta = parsed.choices?.[0]?.delta?.content;
            if (delta) {
              charCount += delta.length;
              yield { type: 'delta', text: delta };
            }
          } catch {
            // ignore partial JSON frames
          }
        }
      }

      yield {
        type: 'done',
        tokensUsed: Math.ceil(charCount / 4),
        latencyMs: Date.now() - start,
        modelUsed: selected.id,
        providerUsed: selected.provider,
      };
      return;
    }

    if (selected.provider === 'anthropic') {
      const resp = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.ANTHROPIC_API_KEY || '',
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: selected.id,
          max_tokens: 4096,
          system: systemInstruction,
          stream: true,
          messages: params.messages
            .filter((m) => m.role !== 'system')
            .map((m) => ({ role: m.role, content: m.content })),
        }),
        signal: params.signal,
      });

      if (!resp.ok || !resp.body) {
        throw new Error(`Anthropic API returned ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let charCount = 0;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          try {
            const parsed = JSON.parse(trimmed.slice(6));
            if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
              charCount += parsed.delta.text.length;
              yield { type: 'delta', text: parsed.delta.text };
            }
          } catch {
            // ignore non-json events
          }
        }
      }

      yield {
        type: 'done',
        tokensUsed: Math.ceil(charCount / 4),
        latencyMs: Date.now() - start,
        modelUsed: selected.id,
        providerUsed: 'anthropic',
      };
    }
  } catch (err) {
    if (params.signal?.aborted) {
      return;
    }
    const errMsg = formatProviderError(err);
    // Explicit fallback to Gemini 3.1 Flash Lite if a secondary provider failed and Gemini is configured
    if (selected.provider !== 'gemini' && getGeminiClient()) {
      yield {
        type: 'notice',
        message: `${selected.name} encountered an error (${errMsg}). Falling back to Gemini 3.1 Flash Lite.`,
      };
      try {
        yield* tryGemini('gemini-3.1-flash-lite');
        return;
      } catch (fallbackErr) {
        yield {
          type: 'error',
          message: formatProviderError(fallbackErr),
        };
        return;
      }
    }
    yield { type: 'error', message: errMsg };
  }
}

export async function generateStructuredText(params: {
  prompt: string;
  systemInstruction?: string;
  modelId?: string;
  useSearch?: boolean;
  thinkingMode?: ThinkingMode;
}): Promise<{ text: string; citations: CitationSource[]; tokensUsed: number; latencyMs: number }> {
  const ai = getGeminiClient();
  if (!ai) {
    throw new Error('GEMINI_API_KEY is not configured on the server.');
  }
  const start = Date.now();
  const modelId = params.modelId || 'gemini-4-hydrogen';
  const config: Record<string, unknown> = {
    systemInstruction: params.systemInstruction
      ? `${params.systemInstruction}\n${CREATOR_IDENTITY_RULE}`
      : `You are NOVA AI. ${CREATOR_IDENTITY_RULE}`,
  };
  if (modelId === 'gemini-4-hydrogen' && params.thinkingMode) {
    config.thinkingConfig = { thinkingLevel: mapThinkingLevel(params.thinkingMode) };
  }
  if (params.useSearch) {
    config.tools = [{ googleSearch: {} }];
  }

  const response = await ai.models.generateContent({
    model: modelId,
    contents: params.prompt,
    config,
  });

  const citations: CitationSource[] = [];
  const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
  if (Array.isArray(chunks)) {
    const seen = new Set<string>();
    for (const c of chunks) {
      const web = (c as { web?: { uri?: string; title?: string } }).web;
      if (web?.uri && !seen.has(web.uri)) {
        seen.add(web.uri);
        citations.push({
          title: web.title || extractDomain(web.uri),
          url: web.uri,
          domain: extractDomain(web.uri),
        });
      }
    }
  }

  return {
    text: response.text ?? '',
    citations,
    tokensUsed: response.usageMetadata?.totalTokenCount ?? 0,
    latencyMs: Date.now() - start,
  };
}

export async function generateEmbedding(text: string): Promise<number[] | null> {
  const ai = getGeminiClient();
  if (!ai) return null;
  try {
    const result = await ai.models.embedContent({
      model: 'gemini-embedding-2',
      contents: text.slice(0, 8000),
    });
    return result.embeddings?.[0]?.values ?? null;
  } catch (err) {
    console.warn('Embedding generation fell back to lexical scoring:', err);
    return null;
  }
}

export async function synthesizeSpeechWav(text: string, voiceName = 'Kore'): Promise<{ audioBase64: string; mimeType: string }> {
  const ai = getGeminiBetaClient();
  if (!ai) {
    throw new Error('GEMINI_API_KEY is required for speech synthesis.');
  }
  const allowedVoices = ['Puck', 'Charon', 'Kore', 'Fenrir', 'Zephyr'];
  const selectedVoice = allowedVoices.includes(voiceName) ? voiceName : 'Kore';

  const response = await ai.models.generateContent({
    model: 'gemini-3.8-flash-lite-tts',
    contents: [
      {
        role: 'user',
        parts: [{ text: text.slice(0, 4000) }],
      },
    ],
    config: {
      responseModalities: ['AUDIO'],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: selectedVoice },
        },
      },
    },
  });

  const part = response.candidates?.[0]?.content?.parts?.[0];
  const base64Audio = part?.inlineData?.data;
  if (!base64Audio) {
    throw new Error('Speech model returned no audio data.');
  }
  return {
    audioBase64: base64Audio,
    mimeType: part?.inlineData?.mimeType || 'audio/wav',
  };
}

export async function transcribeAudioBuffer(base64Audio: string, mimeType: string): Promise<string> {
  const ai = getGeminiClient();
  if (!ai) {
    throw new Error('GEMINI_API_KEY is required for audio transcription.');
  }
  const response = await ai.models.generateContent({
    model: 'gemini-3.5-transcribe',
    contents: {
      parts: [
        {
          inlineData: {
            mimeType,
            data: base64Audio,
          },
        },
        { text: 'Transcribe this audio accurately in its original language (English, Hindi, or Hinglish).' },
      ],
    },
  });
  return response.text?.trim() || '';
}

export async function enhanceImagePrompt(prompt: string, style: string): Promise<string> {
  const ai = getGeminiClient();
  if (!ai) return prompt;
  const response = await ai.models.generateContent({
    model: 'gemini-3.1-flash-lite',
    contents: `Enhance this visual prompt for an image studio with concrete lighting, composition, lens/material detail, and ${style} aesthetic. Return ONLY the enhanced prompt text in 1-2 sentences without quotes or markdown.\n\nOriginal prompt: ${prompt}`,
  });
  return response.text?.trim() || prompt;
}

export async function generateStudioVisual(params: {
  prompt: string;
  aspectRatio: string;
  quality: string;
  style: string;
  referenceImageBase64?: string;
  referenceImageMimeType?: string;
}): Promise<{ dataUrl: string; mimeType: string; provider: string; enhancedPrompt: string }> {
  const enhancedPrompt = await enhanceImagePrompt(params.prompt, params.style);

  // 1. Real OpenAI Raster Image API when OPENAI_API_KEY is configured
  if (process.env.OPENAI_API_KEY && !params.referenceImageBase64) {
    const sizeMap: Record<string, string> = {
      '1:1': '1024x1024',
      '16:9': '1792x1024',
      '9:16': '1024x1792',
      '4:3': '1024x1024',
      '3:4': '1024x1024',
    };
    const resp = await fetch('https://api.openai.com/v1/images/generations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'dall-e-3',
        prompt: enhancedPrompt,
        size: sizeMap[params.aspectRatio] || '1024x1024',
        quality: params.quality === 'hd' ? 'hd' : 'standard',
        response_format: 'b64_json',
      }),
    });
    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`OpenAI Image API error: ${errText}`);
    }
    const json = (await resp.json()) as { data?: Array<{ b64_json?: string }> };
    const b64 = json.data?.[0]?.b64_json;
    if (!b64) throw new Error('OpenAI Image API returned empty image data.');
    return {
      dataUrl: `data:image/png;base64,${b64}`,
      mimeType: 'image/png',
      provider: 'openai-dall-e-3',
      enhancedPrompt,
    };
  }

  // 2. Real Gemini Paid Image Generation ONLY if explicitly enabled via ENABLE_PAID_GEMINI_IMAGE=true
  if (process.env.ENABLE_PAID_GEMINI_IMAGE === 'true' && getGeminiClient()) {
    const ai = getGeminiClient()!;
    const parts: Array<{ text: string } | { inlineData: { data: string; mimeType: string } }> = [];
    if (params.referenceImageBase64 && params.referenceImageMimeType) {
      parts.push({
        inlineData: {
          data: params.referenceImageBase64,
          mimeType: params.referenceImageMimeType,
        },
      });
    }
    parts.push({ text: enhancedPrompt });
    const response = await ai.models.generateContent({
      model: 'gemini-3.1-flash-lite-image',
      contents: { parts },
      config: {
        imageConfig: {
          aspectRatio: params.aspectRatio,
        },
      },
    });
    for (const part of response.candidates?.[0]?.content?.parts ?? []) {
      if (part.inlineData?.data) {
        const mime = part.inlineData.mimeType ?? 'image/png';
        return {
          dataUrl: `data:${mime};base64,${part.inlineData.data}`,
          mimeType: mime,
          provider: 'gemini-3.1-flash-lite-image',
          enhancedPrompt,
        };
      }
    }
  }

  // 3. Free-tier Real AI Vector & SVG Illustration Studio powered by Gemini 4 Hydrogen
  const ai = getGeminiClient();
  if (!ai) {
    throw new Error('No image or Gemini API key configured.');
  }

  const ratioDims: Record<string, { w: number; h: number }> = {
    '1:1': { w: 1024, h: 1024 },
    '16:9': { w: 1280, h: 720 },
    '9:16': { w: 720, h: 1280 },
    '4:3': { w: 1200, h: 900 },
    '3:4': { w: 900, h: 1200 },
  };
  const dims = ratioDims[params.aspectRatio] || ratioDims['1:1'];

  const parts: Array<{ text: string } | { inlineData: { data: string; mimeType: string } }> = [];
  if (params.referenceImageBase64 && params.referenceImageMimeType) {
    parts.push({
      inlineData: {
        data: params.referenceImageBase64,
        mimeType: params.referenceImageMimeType,
      },
    });
  }
  parts.push({
    text: `Create a high-craft, gallery-grade, richly layered SVG illustration (${dims.w}x${dims.h} viewBox="0 0 ${dims.w} ${dims.h}") depicting: ${enhancedPrompt}.
Style: ${params.style}.
Requirements:
- Return ONLY valid standalone <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${dims.w} ${dims.h}" width="${dims.w}" height="${dims.h}">...</svg> markup.
- Use rich <defs> with linearGradient, radialGradient, clipPath, and subtle atmospheric filters/glows.
- Compose detailed geometric, architectural, or organic paths—never a minimal stub.
- Do NOT wrap in markdown code fences or include any text outside the <svg> tag.`,
  });

  const response = await ai.models.generateContent({
    model: 'gemini-4-hydrogen',
    contents: { parts },
    config: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          svgMarkup: {
            type: Type.STRING,
            description: 'Complete valid standalone <svg>...</svg> XML markup.',
          },
        },
        required: ['svgMarkup'],
      },
    },
  });

  const parsed = JSON.parse(response.text || '{}') as { svgMarkup?: string };
  let svg = (parsed.svgMarkup || '').trim();
  if (!svg.startsWith('<svg')) {
    throw new Error('Vector studio returned invalid SVG output. Please refine your prompt.');
  }
  // Strip any script tags for safety
  svg = svg.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
  const base64Svg = Buffer.from(svg, 'utf-8').toString('base64');

  return {
    dataUrl: `data:image/svg+xml;base64,${base64Svg}`,
    mimeType: 'image/svg+xml',
    provider: 'gemini-4-hydrogen-vector-studio',
    enhancedPrompt,
  };
}
