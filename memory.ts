import { Type } from '@google/genai';
import { db, newId } from '../../db/index.ts';
import { generateEmbedding, getGeminiClient } from './providers.ts';
import { cosineSimilarity, lexicalScore } from './rag.ts';
import type { UserMemory } from '../../../src/types/nova.ts';

export async function retrieveRelevantMemories(userId: string, query: string, limit = 5): Promise<UserMemory[]> {
  const pref = db.prepare('SELECT memory_enabled FROM user_preferences WHERE user_id = ?').get(userId) as
    | { memory_enabled: number }
    | undefined;
  if (pref && pref.memory_enabled === 0) {
    return [];
  }

  const rows = db
    .prepare('SELECT id, user_id, category, content, embedding_json, created_at, updated_at FROM user_memories WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100')
    .all(userId) as Array<{
    id: string;
    user_id: string;
    category: UserMemory['category'];
    content: string;
    embedding_json: string | null;
    created_at: string;
    updated_at: string;
  }>;

  if (rows.length === 0) return [];
  if (rows.length <= limit) {
    return rows.map((row) => ({
      id: row.id,
      userId: row.user_id,
      category: row.category,
      content: row.content,
      relevanceScore: 1,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    }));
  }

  const queryVec = await generateEmbedding(query);

  const scored = rows.map((row) => {
    let score = lexicalScore(query, row.content) * 0.35;
    if (queryVec && row.embedding_json) {
      try {
        const vec = JSON.parse(row.embedding_json) as number[];
        score += cosineSimilarity(queryVec, vec) * 0.65;
      } catch {
        // ignore parse errors
      }
    }
    return {
      id: row.id,
      userId: row.user_id,
      category: row.category,
      content: row.content,
      relevanceScore: Number(score.toFixed(3)),
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  });

  scored.sort((a, b) => (b.relevanceScore ?? 0) - (a.relevanceScore ?? 0));
  return scored.slice(0, limit);
}

export async function maybeExtractMemory(userId: string, userMessage: string): Promise<void> {
  const pref = db.prepare('SELECT memory_enabled FROM user_preferences WHERE user_id = ?').get(userId) as
    | { memory_enabled: number }
    | undefined;
  if (pref && pref.memory_enabled === 0) return;

  // Conservative rule: only evaluate messages that contain personal preferences, project info, or explicit "remember" cues
  const cueRegex = /\b(remember|my name|i prefer|i work|my project|i am building|always use|call me|mera naam|mujhe pasand)\b/i;
  if (!cueRegex.test(userMessage) || userMessage.length < 16) return;

  const ai = getGeminiClient();
  if (!ai) return;

  try {
    const resp = await ai.models.generateContent({
      model: 'gemini-3.1-flash-lite',
      contents: `Analyze this user message and extract ONLY durable, factual long-term memories (preferences, ongoing projects, persistent instructions, or biographical facts). If nothing worth saving permanently exists, return an empty array.\n\nUser message: "${userMessage.slice(0, 1000)}"`,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            memories: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  category: { type: Type.STRING, description: 'One of: preference, project, fact, instruction' },
                  content: { type: Type.STRING, description: 'Concise standalone memory statement.' },
                },
                required: ['category', 'content'],
              },
            },
          },
          required: ['memories'],
        },
      },
    });

    const parsed = JSON.parse(resp.text || '{}') as {
      memories?: Array<{ category: string; content: string }>;
    };

    for (const mem of parsed.memories ?? []) {
      const cat = ['preference', 'project', 'fact', 'instruction'].includes(mem.category)
        ? mem.category
        : 'fact';
      const content = mem.content.trim();
      if (!content || content.length < 8) continue;

      // Avoid duplicates
      const existing = db
        .prepare('SELECT id FROM user_memories WHERE user_id = ? AND lower(content) = lower(?)')
        .get(userId, content);
      if (existing) continue;

      const embedding = await generateEmbedding(content);
      db.prepare(
        'INSERT INTO user_memories (id, user_id, category, content, embedding_json) VALUES (?, ?, ?, ?, ?)'
      ).run(newId(), userId, cat, content, embedding ? JSON.stringify(embedding) : null);
    }
  } catch (err) {
    console.warn('Memory extraction skipped:', err);
  }
}
