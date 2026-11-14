import { DatabaseSync } from 'node:sqlite';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const dbPath = process.env.NOVA_DB_PATH || path.resolve(process.cwd(), 'data', 'nova.db');
fs.mkdirSync(path.dirname(dbPath), { recursive: true });

export const db = new DatabaseSync(dbPath);

// Enable WAL and foreign key constraints for reliability and data integrity
db.exec('PRAGMA journal_mode = WAL;');
db.exec('PRAGMA foreign_keys = ON;');

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    plan TEXT NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'premium')),
    avatar_url TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    theme TEXT NOT NULL DEFAULT 'dark',
    default_model TEXT NOT NULL DEFAULT 'gemini-3.1-flash-lite',
    language TEXT NOT NULL DEFAULT 'en',
    memory_enabled INTEGER NOT NULL DEFAULT 1,
    thinking_mode TEXT NOT NULL DEFAULT 'low',
    voice_name TEXT NOT NULL DEFAULT 'Kore',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Conversation',
    model_id TEXT NOT NULL DEFAULT 'gemini-3.1-flash-lite',
    language TEXT NOT NULL DEFAULT 'en',
    research_mode INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);

  CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    model_id TEXT,
    provider TEXT,
    citations_json TEXT NOT NULL DEFAULT '[]',
    attachments_json TEXT NOT NULL DEFAULT '[]',
    tokens_used INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    feedback TEXT CHECK (feedback IN ('up', 'down')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at ASC);

  CREATE TABLE IF NOT EXISTS uploaded_files (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    extracted_text TEXT NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_uploaded_files_user ON uploaded_files(user_id, created_at DESC);

  CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL REFERENCES uploaded_files(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    section_label TEXT,
    content TEXT NOT NULL,
    embedding_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_document_chunks_file ON document_chunks(file_id);

  CREATE TABLE IF NOT EXISTS generated_images (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    enhanced_prompt TEXT,
    provider TEXT NOT NULL,
    aspect_ratio TEXT NOT NULL DEFAULT '1:1',
    quality TEXT NOT NULL DEFAULT 'standard',
    mime_type TEXT NOT NULL DEFAULT 'image/svg+xml',
    data_url TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_generated_images_user ON generated_images(user_id, created_at DESC);

  CREATE TABLE IF NOT EXISTS user_memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('preference', 'project', 'fact', 'instruction')),
    content TEXT NOT NULL,
    embedding_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_user_memories_user ON user_memories(user_id, updated_at DESC);

  CREATE TABLE IF NOT EXISTS coding_projects (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    files_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_coding_projects_user ON coding_projects(user_id, updated_at DESC);

  CREATE TABLE IF NOT EXISTS ai_usage (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ok',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_ai_usage_user ON ai_usage(user_id, created_at DESC);

  CREATE TABLE IF NOT EXISTS user_feedback (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS error_logs (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    route TEXT NOT NULL,
    error_message TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'error',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS password_resets (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS admin_audit_logs (
    id TEXT PRIMARY KEY,
    admin_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS guest_usage (
    guest_id TEXT PRIMARY KEY,
    turns_used INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
`);

export function newId(): string {
  return crypto.randomUUID();
}

export function logSystemError(route: string, errorMessage: string, userId?: string | null, severity: 'warn' | 'error' = 'error') {
  try {
    const stmt = db.prepare(
      'INSERT INTO error_logs (id, user_id, route, error_message, severity) VALUES (?, ?, ?, ?, ?)'
    );
    stmt.run(newId(), userId ?? null, route, errorMessage.slice(0, 1000), severity);
  } catch (err) {
    console.error('Failed to write error log:', err);
  }
}

export function recordAiUsage(params: {
  userId: string;
  modelId: string;
  provider: string;
  endpoint: string;
  tokensUsed: number;
  latencyMs: number;
  status?: 'ok' | 'error';
}) {
  try {
    db.prepare(
      'INSERT INTO ai_usage (id, user_id, model_id, provider, endpoint, tokens_used, latency_ms, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
    ).run(
      newId(),
      params.userId,
      params.modelId,
      params.provider,
      params.endpoint,
      params.tokensUsed,
      params.latencyMs,
      params.status ?? 'ok'
    );
  } catch (err) {
    console.error('Failed to record AI usage:', err);
  }
}

// Seed default system settings if empty
const existingSettings = db.prepare('SELECT key FROM system_settings WHERE key = ?').get('platform');
if (!existingSettings) {
  db.prepare('INSERT INTO system_settings (key, value_json) VALUES (?, ?)').run(
    'platform',
    JSON.stringify({
      allowRegistrations: true,
      defaultModel: 'gemini-3.1-flash-lite',
      freeDailyMessages: 50,
      proDailyMessages: 500,
      premiumDailyMessages: 5000,
      freeMonthlyImages: 15,
      proMonthlyImages: 200,
      premiumMonthlyImages: 2000,
      maintenanceNotice: '',
    })
  );
}
