import 'dotenv/config';
import express from 'express';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';
import crypto from 'node:crypto';
import helmet from 'helmet';
import cookieParser from 'cookie-parser';
import rateLimit from 'express-rate-limit';
import multer from 'multer';
import { WebSocketServer, WebSocket } from 'ws';
import { Modality, type LiveServerMessage } from '@google/genai';
import { createServer as createViteServer } from 'vite';

import { db, logSystemError, newId, recordAiUsage } from './server/db/index.ts';
import {
  clearAuthCookie,
  comparePassword,
  getUserById,
  hashPassword,
  requireAdmin,
  requireAuth,
  setAuthCookie,
  signAuthToken,
  verifyAuthToken,
  type AuthenticatedRequest,
} from './server/lib/auth.ts';
import {
  buildLiveVoiceSystemInstruction,
  generateEmbedding,
  generateStructuredText,
  generateStudioVisual,
  getAvailableModels,
  getGeminiBetaClient,
  getGeminiClient,
  getProviderStatuses,
  streamUnifiedChat,
  synthesizeSpeechWav,
  transcribeAudioBuffer,
} from './server/lib/ai/providers.ts';
import { extractDocumentText, indexDocumentChunks, retrieveRelevantChunks } from './server/lib/ai/rag.ts';
import { maybeExtractMemory, retrieveRelevantMemories } from './server/lib/ai/memory.ts';
import { AI_TOOLS_CATALOG } from './server/lib/tools/catalog.ts';
import type {
  AttachmentPayload,
  LanguagePreference,
  PlanDefinition,
  PlanTier,
  ThinkingMode,
  UserPreferences,
  UserRole,
} from './src/types/nova.ts';

const PORT = Number(process.env.PORT) || 3000;
const isProd = process.env.NODE_ENV === 'production';

const PLANS: PlanDefinition[] = [
  {
    id: 'free',
    name: 'Free',
    tagline: 'For personal exploration and daily productivity',
    priceMonthlyInr: 0,
    priceMonthlyUsd: 0,
    dailyMessageLimit: 50,
    monthlyImageLimit: 15,
    maxDocumentMb: 10,
    modelsAllowed: ['gemini-4-hydrogen', 'gemini-3.1-flash-lite', 'gpt-4o-mini', 'claude-3-5-haiku-latest'],
    features: [
      '50 AI chat turns per day',
      'Live Google Search web citations',
      '15 Studio visuals per month',
      '10 MB document uploads with RAG Q&A',
      'Private long-term memory & Voice AI',
    ],
  },
  {
    id: 'pro',
    name: 'Pro',
    tagline: 'For builders, researchers, and power creators',
    priceMonthlyInr: 1499,
    priceMonthlyUsd: 19,
    dailyMessageLimit: 500,
    monthlyImageLimit: 200,
    maxDocumentMb: 25,
    modelsAllowed: ['gemini-4-hydrogen', 'gemini-3.1-flash-lite', 'gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest'],
    features: [
      '500 AI chat turns per day',
      'Priority multi-provider model routing',
      '200 Studio visuals per month',
      '25 MB multi-document RAG synthesis',
      'Full Coding Workspace & 15 AI Tools',
    ],
  },
  {
    id: 'premium',
    name: 'Premium',
    tagline: 'For scaling teams and high-throughput engineering',
    priceMonthlyInr: 3999,
    priceMonthlyUsd: 49,
    dailyMessageLimit: 5000,
    monthlyImageLimit: 2000,
    maxDocumentMb: 50,
    modelsAllowed: ['all'],
    features: [
      '5,000 AI chat turns per day',
      '2,000 Studio visuals per month',
      'All configured flagship models',
      'High-volume RAG indexing & analytics',
      'Priority support & administrative audit logs',
    ],
  },
];

function getPlatformSettings(): Record<string, unknown> {
  const row = db.prepare('SELECT value_json FROM system_settings WHERE key = ?').get('platform') as
    | { value_json: string }
    | undefined;
  if (!row) return {};
  try {
    return JSON.parse(row.value_json);
  } catch {
    return {};
  }
}

function getUserPreferences(userId: string): UserPreferences {
  const row = db
    .prepare(
      'SELECT theme, default_model, language, memory_enabled, thinking_mode, voice_name FROM user_preferences WHERE user_id = ?'
    )
    .get(userId) as
    | {
        theme: 'dark' | 'light';
        default_model: string;
        language: LanguagePreference;
        memory_enabled: number;
        thinking_mode: ThinkingMode;
        voice_name: string;
      }
    | undefined;

  if (!row) {
    db.prepare('INSERT OR IGNORE INTO user_preferences (user_id, default_model, thinking_mode) VALUES (?, ?, ?)').run(
      userId,
      'gemini-3.1-flash-lite',
      'low'
    );
    return {
      theme: 'dark',
      defaultModel: 'gemini-3.1-flash-lite',
      language: 'en',
      memoryEnabled: true,
      thinkingMode: 'low',
      voiceName: 'Kore',
    };
  }
  return {
    theme: row.theme,
    defaultModel: row.default_model,
    language: row.language,
    memoryEnabled: Boolean(row.memory_enabled),
    thinkingMode: row.thinking_mode,
    voiceName: row.voice_name,
  };
}

function getPlanLimits(plan: PlanTier) {
  const settings = getPlatformSettings();
  if (plan === 'premium') {
    return {
      dailyMessages: Number(settings.premiumDailyMessages) || 5000,
      monthlyImages: Number(settings.premiumMonthlyImages) || 2000,
    };
  }
  if (plan === 'pro') {
    return {
      dailyMessages: Number(settings.proDailyMessages) || 500,
      monthlyImages: Number(settings.proMonthlyImages) || 200,
    };
  }
  return {
    dailyMessages: Number(settings.freeDailyMessages) || 50,
    monthlyImages: Number(settings.freeMonthlyImages) || 15,
  };
}

function getDailyMessageUsage(userId: string): number {
  const row = db
    .prepare(
      "SELECT COUNT(*) as count FROM ai_usage WHERE user_id = ? AND endpoint = 'chat' AND date(created_at) = date('now')"
    )
    .get(userId) as { count: number };
  return row?.count ?? 0;
}

function getMonthlyImageUsage(userId: string): number {
  const row = db
    .prepare(
      "SELECT COUNT(*) as count FROM generated_images WHERE user_id = ? AND strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')"
    )
    .get(userId) as { count: number };
  return row?.count ?? 0;
}

async function startServer() {
  const app = express();
  app.set('trust proxy', 1);

  app.use(
    helmet({
      contentSecurityPolicy: false,
      crossOriginEmbedderPolicy: false,
    })
  );
  app.use(cookieParser());
  app.use(express.json({ limit: '16mb' }));
  app.use(express.urlencoded({ extended: true, limit: '16mb' }));

  const apiLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 180,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many requests. Please slow down briefly.' },
  });
  app.use('/api', apiLimiter);

  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 15 * 1024 * 1024 },
  });

  // ==========================================
  // PUBLIC HEALTH, MODELS, PLANS
  // ==========================================
  app.get('/api/health', (_req, res) => {
    const models = getAvailableModels();
    res.json({
      status: 'ok',
      service: 'NOVA AI',
      configuredModelsCount: models.length,
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/api/models', (_req, res) => {
    res.json({
      models: getAvailableModels(),
      providers: getProviderStatuses().map((p) => ({
        id: p.id,
        name: p.name,
        configured: p.configured,
        models: p.models,
      })),
    });
  });

  app.get('/api/subscriptions/plans', (_req, res) => {
    res.json({
      plans: PLANS,
      gateways: {
        razorpayConfigured: Boolean(process.env.RAZORPAY_KEY_ID && process.env.RAZORPAY_KEY_SECRET),
        stripeConfigured: Boolean(process.env.STRIPE_SECRET_KEY),
      },
    });
  });

  // ==========================================
  // AUTHENTICATION & ACCOUNT ROUTES
  // ==========================================
  app.post('/api/auth/register', async (req, res) => {
    try {
      const settings = getPlatformSettings();
      if (settings.allowRegistrations === false) {
        return res.status(403).json({ error: 'New registrations are currently paused by the administrator.' });
      }
      const { email, password, name } = req.body as { email?: string; password?: string; name?: string };
      const normalizedEmail = (email || '').trim().toLowerCase();
      const trimmedName = (name || '').trim();

      if (!normalizedEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
        return res.status(400).json({ error: 'Please enter a valid email address.' });
      }
      if (!trimmedName || trimmedName.length < 2) {
        return res.status(400).json({ error: 'Name must be at least 2 characters.' });
      }
      if (!password || password.length < 8) {
        return res.status(400).json({ error: 'Password must be at least 8 characters.' });
      }

      const existing = db.prepare('SELECT id FROM users WHERE email = ?').get(normalizedEmail);
      if (existing) {
        return res.status(409).json({ error: 'An account with this email already exists.' });
      }

      const userCountRow = db.prepare('SELECT COUNT(*) as c FROM users').get() as { c: number };
      const adminEmail = (process.env.ADMIN_EMAIL || 'savohere1@gmail.com').trim().toLowerCase();
      const role: UserRole = userCountRow.c === 0 || normalizedEmail === adminEmail ? 'admin' : 'user';

      const id = newId();
      const passwordHash = await hashPassword(password);
      db.prepare('INSERT INTO users (id, email, name, password_hash, role, plan) VALUES (?, ?, ?, ?, ?, ?)').run(
        id,
        normalizedEmail,
        trimmedName,
        passwordHash,
        role,
        'free'
      );
      db.prepare('INSERT INTO user_preferences (user_id, default_model, thinking_mode) VALUES (?, ?, ?)').run(
        id,
        'gemini-3.1-flash-lite',
        'low'
      );

      const user = getUserById(id)!;
      const preferences = getUserPreferences(id);
      const token = signAuthToken({ id: user.id, email: user.email, role: user.role });
      setAuthCookie(res, token);

      res.status(201).json({ user, preferences, token });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Registration failed';
      logSystemError('/api/auth/register', msg);
      res.status(500).json({ error: 'Unable to complete registration.' });
    }
  });

  app.post('/api/auth/login', async (req, res) => {
    try {
      const { email, password } = req.body as { email?: string; password?: string };
      const normalizedEmail = (email || '').trim().toLowerCase();
      if (!normalizedEmail || !password) {
        return res.status(400).json({ error: 'Email and password are required.' });
      }

      const row = db
        .prepare('SELECT id, email, password_hash, role FROM users WHERE email = ?')
        .get(normalizedEmail) as { id: string; email: string; password_hash: string; role: UserRole } | undefined;

      if (!row || !(await comparePassword(password, row.password_hash))) {
        return res.status(401).json({ error: 'Invalid email or password.' });
      }

      const user = getUserById(row.id)!;
      const preferences = getUserPreferences(row.id);
      const token = signAuthToken({ id: user.id, email: user.email, role: user.role });
      setAuthCookie(res, token);

      res.json({ user, preferences, token });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Login failed';
      logSystemError('/api/auth/login', msg);
      res.status(500).json({ error: 'Unable to sign in right now.' });
    }
  });

  app.post('/api/auth/logout', (_req, res) => {
    clearAuthCookie(res);
    res.json({ success: true });
  });

  app.post('/api/auth/forgot-password', (req, res) => {
    const { email } = req.body as { email?: string };
    const normalizedEmail = (email || '').trim().toLowerCase();
    if (!normalizedEmail) {
      return res.status(400).json({ error: 'Email is required.' });
    }
    const row = db.prepare('SELECT id FROM users WHERE email = ?').get(normalizedEmail) as { id: string } | undefined;
    if (!row) {
      return res.json({
        message: 'If an account matches that email, a password reset token has been generated.',
      });
    }
    const token = crypto.randomBytes(24).toString('hex');
    const expiresAt = new Date(Date.now() + 30 * 60 * 1000).toISOString();
    db.prepare('INSERT INTO password_resets (token, user_id, expires_at) VALUES (?, ?, ?)').run(
      token,
      row.id,
      expiresAt
    );
    res.json({
      message: 'Password reset token generated. Enter this token below to set a new password (expires in 30 minutes).',
      resetToken: token,
    });
  });

  app.post('/api/auth/reset-password', async (req, res) => {
    const { token, newPassword } = req.body as { token?: string; newPassword?: string };
    if (!token || !newPassword || newPassword.length < 8) {
      return res.status(400).json({ error: 'Valid reset token and new password (min 8 chars) are required.' });
    }
    const row = db
      .prepare('SELECT token, user_id, expires_at, used FROM password_resets WHERE token = ?')
      .get(token) as { token: string; user_id: string; expires_at: string; used: number } | undefined;

    if (!row || row.used || new Date(row.expires_at).getTime() < Date.now()) {
      return res.status(400).json({ error: 'Reset token is invalid or expired.' });
    }
    const passwordHash = await hashPassword(newPassword);
    db.prepare("UPDATE users SET password_hash = ?, updated_at = datetime('now') WHERE id = ?").run(
      passwordHash,
      row.user_id
    );
    db.prepare('UPDATE password_resets SET used = 1 WHERE token = ?').run(token);
    res.json({ message: 'Password updated successfully. You may now sign in.' });
  });

  app.get('/api/auth/me', requireAuth, (req: AuthenticatedRequest, res) => {
    const preferences = getUserPreferences(req.user!.id);
    res.json({ user: req.user, preferences });
  });

  app.patch('/api/profile', requireAuth, async (req: AuthenticatedRequest, res) => {
    const { name, currentPassword, newPassword } = req.body as {
      name?: string;
      currentPassword?: string;
      newPassword?: string;
    };
    const userId = req.user!.id;

    if (name && name.trim().length >= 2) {
      db.prepare("UPDATE users SET name = ?, updated_at = datetime('now') WHERE id = ?").run(name.trim(), userId);
    }

    if (newPassword) {
      if (newPassword.length < 8) {
        return res.status(400).json({ error: 'New password must be at least 8 characters.' });
      }
      const row = db.prepare('SELECT password_hash FROM users WHERE id = ?').get(userId) as { password_hash: string };
      if (!currentPassword || !(await comparePassword(currentPassword, row.password_hash))) {
        return res.status(400).json({ error: 'Current password is incorrect.' });
      }
      const nextHash = await hashPassword(newPassword);
      db.prepare("UPDATE users SET password_hash = ?, updated_at = datetime('now') WHERE id = ?").run(nextHash, userId);
    }

    res.json({ user: getUserById(userId) });
  });

  app.patch('/api/preferences', requireAuth, (req: AuthenticatedRequest, res) => {
    const userId = req.user!.id;
    const current = getUserPreferences(userId);
    const body = req.body as Partial<UserPreferences>;

    const next: UserPreferences = {
      theme: body.theme === 'light' || body.theme === 'dark' ? body.theme : current.theme,
      defaultModel: typeof body.defaultModel === 'string' && body.defaultModel ? body.defaultModel : current.defaultModel,
      language: typeof body.language === 'string' && body.language.trim() ? body.language.trim().slice(0, 32) : current.language,
      memoryEnabled: typeof body.memoryEnabled === 'boolean' ? body.memoryEnabled : current.memoryEnabled,
      thinkingMode: body.thinkingMode === 'low' || body.thinkingMode === 'medium' || body.thinkingMode === 'high' ? body.thinkingMode : current.thinkingMode,
      voiceName: typeof body.voiceName === 'string' && body.voiceName ? body.voiceName : current.voiceName,
    };

    db.prepare(
      `INSERT INTO user_preferences (user_id, theme, default_model, language, memory_enabled, thinking_mode, voice_name, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
       ON CONFLICT(user_id) DO UPDATE SET
         theme = excluded.theme,
         default_model = excluded.default_model,
         language = excluded.language,
         memory_enabled = excluded.memory_enabled,
         thinking_mode = excluded.thinking_mode,
         voice_name = excluded.voice_name,
         updated_at = datetime('now')`
    ).run(
      userId,
      next.theme,
      next.defaultModel,
      next.language,
      next.memoryEnabled ? 1 : 0,
      next.thinkingMode,
      next.voiceName
    );

    res.json({ preferences: next });
  });

  // ==========================================
  // CONVERSATIONS & CHAT STREAMING
  // ==========================================
  app.get('/api/conversations', requireAuth, (req: AuthenticatedRequest, res) => {
    const q = ((req.query.q as string) || '').trim();
    const userId = req.user!.id;

    let rows;
    if (q) {
      const like = `%${q}%`;
      rows = db
        .prepare(
          `SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count
           FROM conversations c
           WHERE c.user_id = ? AND (
             c.title LIKE ? OR EXISTS (
               SELECT 1 FROM messages m2 WHERE m2.conversation_id = c.id AND m2.content LIKE ?
             )
           )
           ORDER BY c.updated_at DESC LIMIT 100`
        )
        .all(userId, like, like);
    } else {
      rows = db
        .prepare(
          `SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count
           FROM conversations c
           WHERE c.user_id = ?
           ORDER BY c.updated_at DESC LIMIT 100`
        )
        .all(userId);
    }

    res.json({
      conversations: (rows as Array<Record<string, unknown>>).map((r) => ({
        id: r.id,
        userId: r.user_id,
        title: r.title,
        modelId: r.model_id,
        language: r.language,
        researchMode: Boolean(r.research_mode),
        createdAt: r.created_at,
        updatedAt: r.updated_at,
        messageCount: r.message_count ?? 0,
      })),
    });
  });

  app.post('/api/conversations', requireAuth, (req: AuthenticatedRequest, res) => {
    const userId = req.user!.id;
    const prefs = getUserPreferences(userId);
    const { title, modelId, language, researchMode } = req.body as {
      title?: string;
      modelId?: string;
      language?: LanguagePreference;
      researchMode?: boolean;
    };
    const id = newId();
    db.prepare(
      'INSERT INTO conversations (id, user_id, title, model_id, language, research_mode) VALUES (?, ?, ?, ?, ?, ?)'
    ).run(
      id,
      userId,
      (title || 'New Conversation').trim().slice(0, 120),
      modelId || prefs.defaultModel,
      language || prefs.language,
      researchMode ? 1 : 0
    );

    const row = db.prepare('SELECT * FROM conversations WHERE id = ?').get(id) as Record<string, unknown>;
    res.status(201).json({
      conversation: {
        id: row.id,
        userId: row.user_id,
        title: row.title,
        modelId: row.model_id,
        language: row.language,
        researchMode: Boolean(row.research_mode),
        createdAt: row.created_at,
        updatedAt: row.updated_at,
        messageCount: 0,
      },
    });
  });

  app.patch('/api/conversations/:id', requireAuth, (req: AuthenticatedRequest, res) => {
    const userId = req.user!.id;
    const { id } = req.params;
    const existing = db.prepare('SELECT * FROM conversations WHERE id = ? AND user_id = ?').get(id, userId) as
      | Record<string, unknown>
      | undefined;
    if (!existing) return res.status(404).json({ error: 'Conversation not found.' });

    const { title, modelId, language, researchMode } = req.body as {
      title?: string;
      modelId?: string;
      language?: LanguagePreference;
      researchMode?: boolean;
    };

    db.prepare(
      `UPDATE conversations
       SET title = ?, model_id = ?, language = ?, research_mode = ?, updated_at = datetime('now')
       WHERE id = ? AND user_id = ?`
    ).run(
      title !== undefined ? title.trim().slice(0, 120) : (existing.title as string),
      modelId !== undefined ? modelId : (existing.model_id as string),
      language !== undefined ? language : (existing.language as string),
      researchMode !== undefined ? (researchMode ? 1 : 0) : (existing.research_mode as number),
      id,
      userId
    );

    const updated = db.prepare('SELECT * FROM conversations WHERE id = ?').get(id) as Record<string, unknown>;
    res.json({
      conversation: {
        id: updated.id,
        userId: updated.user_id,
        title: updated.title,
        modelId: updated.model_id,
        language: updated.language,
        researchMode: Boolean(updated.research_mode),
        createdAt: updated.created_at,
        updatedAt: updated.updated_at,
      },
    });
  });

  app.delete('/api/conversations/:id', requireAuth, (req: AuthenticatedRequest, res) => {
    const result = db
      .prepare('DELETE FROM conversations WHERE id = ? AND user_id = ?')
      .run(req.params.id, req.user!.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Conversation not found.' });
    res.json({ success: true });
  });

  app.get('/api/conversations/:id/messages', requireAuth, (req: AuthenticatedRequest, res) => {
    const conv = db
      .prepare('SELECT id FROM conversations WHERE id = ? AND user_id = ?')
      .get(req.params.id, req.user!.id);
    if (!conv) return res.status(404).json({ error: 'Conversation not found.' });

    const rows = db
      .prepare('SELECT * FROM messages WHERE conversation_id = ? AND user_id = ? ORDER BY created_at ASC, rowid ASC')
      .all(req.params.id, req.user!.id) as Array<Record<string, unknown>>;

    res.json({
      messages: rows.map((r) => ({
        id: r.id,
        conversationId: r.conversation_id,
        role: r.role,
        content: r.content,
        modelId: r.model_id,
        provider: r.provider,
        citations: JSON.parse((r.citations_json as string) || '[]'),
        attachments: JSON.parse((r.attachments_json as string) || '[]'),
        tokensUsed: r.tokens_used,
        latencyMs: r.latency_ms,
        feedback: r.feedback,
        createdAt: r.created_at,
      })),
    });
  });

  app.post('/api/chat/stream', requireAuth, async (req: AuthenticatedRequest, res) => {
    const user = req.user!;

    const {
      conversationId,
      message,
      modelId,
      language,
      learningMode,
      thinkingMode,
      researchMode,
      attachments,
      editMessageId,
      regenerateFromAssistantId,
    } = req.body as {
      conversationId?: string;
      message?: string;
      modelId?: string;
      language?: LanguagePreference;
      learningMode?: boolean;
      thinkingMode?: ThinkingMode;
      researchMode?: boolean;
      attachments?: AttachmentPayload[];
      editMessageId?: string;
      regenerateFromAssistantId?: string;
    };

    const limits = getPlanLimits(user.plan);
    const usedToday = getDailyMessageUsage(user.id);
    if (usedToday >= limits.dailyMessages) {
      return res.status(429).json({
        error: `Daily limit of ${limits.dailyMessages} messages reached for your ${user.plan} plan.`,
      });
    }

    const trimmedMsg = (message || '').trim();
    if (!regenerateFromAssistantId && !trimmedMsg && (!attachments || attachments.length === 0)) {
      return res.status(400).json({ error: 'Message content or attachment is required.' });
    }
    if (trimmedMsg.length > 32000) {
      return res.status(400).json({ error: 'Message exceeds maximum length of 32,000 characters.' });
    }

    const prefs = getUserPreferences(user.id);

    let activeConvId = conversationId || '';
    if (!activeConvId) {
      activeConvId = newId();
      const autoTitle = trimmedMsg ? trimmedMsg.slice(0, 56) + (trimmedMsg.length > 56 ? '...' : '') : 'New Conversation';
      db.prepare(
        'INSERT INTO conversations (id, user_id, title, model_id, language, research_mode) VALUES (?, ?, ?, ?, ?, ?)'
      ).run(
        activeConvId,
        user.id,
        autoTitle,
        modelId || prefs.defaultModel,
        language || prefs.language,
        researchMode ? 1 : 0
      );
    } else {
      const owns = db
        .prepare('SELECT id, title FROM conversations WHERE id = ? AND user_id = ?')
        .get(activeConvId, user.id) as { id: string; title: string } | undefined;
      if (!owns) return res.status(404).json({ error: 'Conversation not found.' });
      if (owns.title === 'New Conversation' && trimmedMsg) {
        const autoTitle = trimmedMsg.slice(0, 56) + (trimmedMsg.length > 56 ? '...' : '');
        db.prepare('UPDATE conversations SET title = ? WHERE id = ?').run(autoTitle, activeConvId);
      }
    }

    if (editMessageId) {
      const target = db
        .prepare('SELECT rowid FROM messages WHERE id = ? AND conversation_id = ? AND user_id = ?')
        .get(editMessageId, activeConvId, user.id) as { rowid: number } | undefined;
      if (target) {
        db.prepare('DELETE FROM messages WHERE conversation_id = ? AND user_id = ? AND rowid >= ?').run(
          activeConvId,
          user.id,
          target.rowid
        );
      }
    } else if (regenerateFromAssistantId) {
      const target = db
        .prepare('SELECT rowid FROM messages WHERE id = ? AND conversation_id = ? AND user_id = ?')
        .get(regenerateFromAssistantId, activeConvId, user.id) as { rowid: number } | undefined;
      if (target) {
        db.prepare('DELETE FROM messages WHERE conversation_id = ? AND user_id = ? AND rowid >= ?').run(
          activeConvId,
          user.id,
          target.rowid
        );
      }
    }

    let userMsgId = '';
    if (!regenerateFromAssistantId) {
      userMsgId = newId();
      const sanitizedAttachments = (attachments || []).map((a) => ({
        name: a.name,
        mimeType: a.mimeType,
        size: a.size,
        base64: a.mimeType.startsWith('image/') ? a.base64 : undefined,
        extractedText: a.extractedText,
      }));
      db.prepare(
        `INSERT INTO messages (id, conversation_id, user_id, role, content, attachments_json)
         VALUES (?, ?, ?, 'user', ?, ?)`
      ).run(userMsgId, activeConvId, user.id, trimmedMsg, JSON.stringify(sanitizedAttachments));
    }

    const historyRows = db
      .prepare('SELECT role, content, attachments_json FROM messages WHERE conversation_id = ? AND user_id = ? ORDER BY rowid ASC')
      .all(activeConvId, user.id) as Array<{ role: 'user' | 'assistant' | 'system'; content: string; attachments_json: string }>;

    const contextMessages = historyRows.slice(-20).map((r) => ({
      role: r.role,
      content: r.content,
      attachments: JSON.parse(r.attachments_json || '[]') as AttachmentPayload[],
    }));

    const latestUserPrompt =
      trimmedMsg || contextMessages.filter((m) => m.role === 'user').slice(-1)[0]?.content || '';

    const memories = prefs.memoryEnabled ? await retrieveRelevantMemories(user.id, latestUserPrompt, 5) : [];

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders?.();

    const abortController = new AbortController();
    res.on('close', () => {
      if (!res.writableEnded) {
        abortController.abort();
      }
    });

    const sendEvent = (payload: Record<string, unknown>) => {
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify(payload)}\n\n`);
      }
    };

    sendEvent({ type: 'meta', conversationId: activeConvId, userMessageId: userMsgId });

    let fullText = '';
    let finalCitations: unknown[] = [];
    let tokensUsed = 0;
    let latencyMs = 0;
    let modelUsed = modelId || prefs.defaultModel;
    let providerUsed = 'gemini';

    try {
      for await (const ev of streamUnifiedChat({
        modelId: modelId || prefs.defaultModel,
        messages: contextMessages,
        language: language || prefs.language,
        learningMode: Boolean(learningMode),
        thinkingMode: thinkingMode || prefs.thinkingMode,
        researchMode: Boolean(researchMode),
        memories: memories.map((m) => `[${m.category}] ${m.content}`),
        signal: abortController.signal,
      })) {
        if (ev.type === 'delta' && ev.text) {
          fullText += ev.text;
          sendEvent({ type: 'delta', text: ev.text });
        } else if (ev.type === 'citations' && ev.citations) {
          finalCitations = ev.citations;
          sendEvent({ type: 'citations', citations: ev.citations });
        } else if (ev.type === 'notice' && ev.message) {
          sendEvent({ type: 'notice', message: ev.message });
        } else if (ev.type === 'error' && ev.message) {
          logSystemError('/api/chat/stream', ev.message, user.id);
          sendEvent({ type: 'error', message: ev.message });
          recordAiUsage({
            userId: user.id,
            modelId: modelUsed,
            provider: providerUsed,
            endpoint: 'chat',
            tokensUsed: 0,
            latencyMs: 0,
            status: 'error',
          });
          res.end();
          return;
        } else if (ev.type === 'done') {
          tokensUsed = ev.tokensUsed || Math.ceil(fullText.length / 4);
          latencyMs = ev.latencyMs || 0;
          modelUsed = ev.modelUsed || modelUsed;
          providerUsed = ev.providerUsed || providerUsed;
        }
      }

      const assistantMsgId = newId();
      if (fullText.trim()) {
        db.prepare(
          `INSERT INTO messages (id, conversation_id, user_id, role, content, model_id, provider, citations_json, tokens_used, latency_ms)
           VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?, ?, ?)`
        ).run(
          assistantMsgId,
          activeConvId,
          user.id,
          fullText,
          modelUsed,
          providerUsed,
          JSON.stringify(finalCitations),
          tokensUsed,
          latencyMs
        );
        db.prepare("UPDATE conversations SET updated_at = datetime('now'), model_id = ? WHERE id = ?").run(
          modelUsed,
          activeConvId
        );
        recordAiUsage({
          userId: user.id,
          modelId: modelUsed,
          provider: providerUsed,
          endpoint: 'chat',
          tokensUsed,
          latencyMs,
          status: 'ok',
        });
        if (latestUserPrompt) {
          void maybeExtractMemory(user.id, latestUserPrompt);
        }
      }

      sendEvent({
        type: 'done',
        conversationId: activeConvId,
        assistantMessageId: assistantMsgId,
        modelId: modelUsed,
        provider: providerUsed,
        tokensUsed,
        latencyMs,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Streaming failed';
      logSystemError('/api/chat/stream', msg, user.id);
      sendEvent({ type: 'error', message: msg });
    } finally {
      if (!res.writableEnded) res.end();
    }
  });

  app.post('/api/messages/:id/feedback', requireAuth, (req: AuthenticatedRequest, res) => {
    const { rating, comment } = req.body as { rating?: 'up' | 'down'; comment?: string };
    if (rating !== 'up' && rating !== 'down') {
      return res.status(400).json({ error: 'Rating must be up or down.' });
    }
    const msg = db
      .prepare('SELECT id FROM messages WHERE id = ? AND user_id = ?')
      .get(req.params.id, req.user!.id);
    if (!msg) return res.status(404).json({ error: 'Message not found.' });

    db.prepare('UPDATE messages SET feedback = ? WHERE id = ?').run(rating, req.params.id);
    db.prepare('INSERT INTO user_feedback (id, user_id, message_id, rating, comment) VALUES (?, ?, ?, ?, ?)').run(
      newId(),
      req.user!.id,
      req.params.id,
      rating,
      comment || null
    );
    res.json({ success: true });
  });

  // ==========================================
  // CHAT FILE ATTACHMENT TEXT EXTRACTION
  // ==========================================
  app.post('/api/chat/extract-attachment', requireAuth, upload.single('file'), async (req: AuthenticatedRequest, res) => {
    try {
      if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });
      const { originalname, mimetype, size, buffer } = req.file;

      if (mimetype.startsWith('image/')) {
        return res.json({
          attachment: {
            name: originalname,
            mimeType: mimetype,
            size,
            base64: buffer.toString('base64'),
          },
        });
      }

      const extractedText = await extractDocumentText(buffer, originalname, mimetype);
      res.json({
        attachment: {
          name: originalname,
          mimeType: mimetype,
          size,
          extractedText: extractedText.slice(0, 32000),
        },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to parse attachment';
      logSystemError('/api/chat/extract-attachment', msg, req.user?.id);
      res.status(400).json({ error: msg });
    }
  });

  // ==========================================
  // VOICE AI (TTS & TRANSCRIPTION)
  // ==========================================
  app.post('/api/voice/tts', requireAuth, async (req: AuthenticatedRequest, res) => {
    try {
      const { text, voiceName } = req.body as { text?: string; voiceName?: string };
      if (!text || !text.trim()) return res.status(400).json({ error: 'Text is required for TTS.' });
      const start = Date.now();
      const result = await synthesizeSpeechWav(text.trim(), voiceName);
      recordAiUsage({
        userId: req.user!.id,
        modelId: 'gemini-3.8-flash-lite-tts',
        provider: 'gemini',
        endpoint: 'voice-tts',
        tokensUsed: Math.ceil(text.length / 4),
        latencyMs: Date.now() - start,
      });
      res.json(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'TTS failed';
      logSystemError('/api/voice/tts', msg, req.user?.id);
      res.status(500).json({ error: msg });
    }
  });

  app.post('/api/voice/transcribe', requireAuth, upload.single('audio'), async (req: AuthenticatedRequest, res) => {
    try {
      if (!req.file) return res.status(400).json({ error: 'Audio recording is required.' });
      const start = Date.now();
      const text = await transcribeAudioBuffer(req.file.buffer.toString('base64'), req.file.mimetype || 'audio/webm');
      recordAiUsage({
        userId: req.user!.id,
        modelId: 'gemini-3.5-transcribe',
        provider: 'gemini',
        endpoint: 'voice-stt',
        tokensUsed: Math.ceil(text.length / 4),
        latencyMs: Date.now() - start,
      });
      res.json({ text });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Audio transcription failed';
      logSystemError('/api/voice/transcribe', msg, req.user?.id);
      res.status(500).json({ error: msg });
    }
  });

  // ==========================================
  // IMAGE STUDIO
  // ==========================================
  app.get('/api/images', requireAuth, (req: AuthenticatedRequest, res) => {
    const rows = db
      .prepare('SELECT * FROM generated_images WHERE user_id = ? ORDER BY created_at DESC LIMIT 60')
      .all(req.user!.id) as Array<Record<string, unknown>>;

    res.json({
      images: rows.map((r) => ({
        id: r.id,
        userId: r.user_id,
        prompt: r.prompt,
        enhancedPrompt: r.enhanced_prompt,
        provider: r.provider,
        aspectRatio: r.aspect_ratio,
        quality: r.quality,
        mimeType: r.mime_type,
        dataUrl: r.data_url,
        createdAt: r.created_at,
      })),
      activeMode: process.env.OPENAI_API_KEY
        ? 'openai-dall-e-3'
        : process.env.ENABLE_PAID_GEMINI_IMAGE === 'true'
        ? 'gemini-3.1-flash-lite-image'
        : 'gemini-4-hydrogen-vector-studio',
    });
  });

  app.post('/api/images/generate', requireAuth, async (req: AuthenticatedRequest, res) => {
    try {
      const user = req.user!;
      const limits = getPlanLimits(user.plan);
      if (getMonthlyImageUsage(user.id) >= limits.monthlyImages) {
        return res.status(429).json({
          error: `Monthly studio quota of ${limits.monthlyImages} visuals reached on your ${user.plan} plan.`,
        });
      }

      const { prompt, aspectRatio, quality, style, referenceImageBase64, referenceImageMimeType } = req.body as {
        prompt?: string;
        aspectRatio?: string;
        quality?: string;
        style?: string;
        referenceImageBase64?: string;
        referenceImageMimeType?: string;
      };

      if (!prompt || prompt.trim().length < 3) {
        return res.status(400).json({ error: 'Please describe the visual you want to generate.' });
      }

      const start = Date.now();
      const result = await generateStudioVisual({
        prompt: prompt.trim(),
        aspectRatio: aspectRatio || '1:1',
        quality: quality || 'standard',
        style: style || 'Futuristic Architectural',
        referenceImageBase64,
        referenceImageMimeType,
      });

      const id = newId();
      db.prepare(
        `INSERT INTO generated_images (id, user_id, prompt, enhanced_prompt, provider, aspect_ratio, quality, mime_type, data_url)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        id,
        user.id,
        prompt.trim(),
        result.enhancedPrompt,
        result.provider,
        aspectRatio || '1:1',
        quality || 'standard',
        result.mimeType,
        result.dataUrl
      );

      recordAiUsage({
        userId: user.id,
        modelId: result.provider,
        provider: result.provider.startsWith('openai') ? 'openai' : 'gemini',
        endpoint: 'image-studio',
        tokensUsed: 500,
        latencyMs: Date.now() - start,
      });

      const created = db.prepare('SELECT * FROM generated_images WHERE id = ?').get(id) as Record<string, unknown>;
      res.status(201).json({
        image: {
          id: created.id,
          userId: created.user_id,
          prompt: created.prompt,
          enhancedPrompt: created.enhanced_prompt,
          provider: created.provider,
          aspectRatio: created.aspect_ratio,
          quality: created.quality,
          mimeType: created.mime_type,
          dataUrl: created.data_url,
          createdAt: created.created_at,
        },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Image generation failed';
      logSystemError('/api/images/generate', msg, req.user!.id);
      res.status(500).json({ error: msg });
    }
  });

  app.delete('/api/images/:id', requireAuth, (req: AuthenticatedRequest, res) => {
    const result = db
      .prepare('DELETE FROM generated_images WHERE id = ? AND user_id = ?')
      .run(req.params.id, req.user!.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Image not found.' });
    res.json({ success: true });
  });

  // ==========================================
  // DOCUMENT INTELLIGENCE & RAG
  // ==========================================
  app.get('/api/documents', requireAuth, (req: AuthenticatedRequest, res) => {
    const rows = db
      .prepare(
        `SELECT uf.id, uf.user_id, uf.filename, uf.mime_type, uf.size_bytes, uf.summary, uf.created_at,
                (SELECT COUNT(*) FROM document_chunks dc WHERE dc.file_id = uf.id) as chunk_count
         FROM uploaded_files uf
         WHERE uf.user_id = ?
         ORDER BY uf.created_at DESC`
      )
      .all(req.user!.id) as Array<Record<string, unknown>>;

    res.json({
      documents: rows.map((r) => ({
        id: r.id,
        userId: r.user_id,
        filename: r.filename,
        mimeType: r.mime_type,
        sizeBytes: r.size_bytes,
        summary: r.summary,
        chunkCount: r.chunk_count,
        createdAt: r.created_at,
      })),
    });
  });

  app.post('/api/documents/upload', requireAuth, upload.single('file'), async (req: AuthenticatedRequest, res) => {
    try {
      if (!req.file) return res.status(400).json({ error: 'Please select a file to upload.' });
      const { originalname, mimetype, size, buffer } = req.file;
      const userId = req.user!.id;

      const extractedText = await extractDocumentText(buffer, originalname, mimetype);
      if (!extractedText) {
        return res.status(400).json({ error: 'No readable text could be extracted from this file.' });
      }

      let summary = '';
      try {
        const sumRes = await generateStructuredText({
          modelId: 'gemini-3.1-flash-lite',
          prompt: `Provide a crisp 3-5 bullet executive summary of this document (${originalname}). Base it strictly on the text:\n\n${extractedText.slice(0, 20000)}`,
        });
        summary = sumRes.text;
      } catch {
        summary = extractedText.slice(0, 320) + (extractedText.length > 320 ? '...' : '');
      }

      const fileId = newId();
      db.prepare(
        'INSERT INTO uploaded_files (id, user_id, filename, mime_type, size_bytes, extracted_text, summary) VALUES (?, ?, ?, ?, ?, ?, ?)'
      ).run(fileId, userId, originalname, mimetype, size, extractedText, summary);

      const chunkCount = await indexDocumentChunks({ fileId, userId, text: extractedText });

      res.status(201).json({
        document: {
          id: fileId,
          userId,
          filename: originalname,
          mimeType: mimetype,
          sizeBytes: size,
          summary,
          chunkCount,
          createdAt: new Date().toISOString(),
        },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Document ingestion failed';
      logSystemError('/api/documents/upload', msg, req.user!.id);
      res.status(500).json({ error: msg });
    }
  });

  app.post('/api/documents/query', requireAuth, async (req: AuthenticatedRequest, res) => {
    try {
      const userId = req.user!.id;
      const { fileIds, question, mode } = req.body as {
        fileIds?: string[];
        question?: string;
        mode?: 'qa' | 'report' | 'compare' | 'spreadsheet';
      };
      if (!fileIds || fileIds.length === 0) {
        return res.status(400).json({ error: 'Select at least one document to analyze.' });
      }
      const queryText = (question || 'Summarize key findings, metrics, and risks.').trim();
      const chunks = await retrieveRelevantChunks({ userId, fileIds, query: queryText, topK: 8 });
      if (chunks.length === 0) {
        return res.status(404).json({ error: 'No indexed document sections found for the selected files.' });
      }

      const contextBlock = chunks
        .map((c, idx) => `[Source ${idx + 1} | ${c.filename} | ${c.sectionLabel}]\n${c.content}`)
        .join('\n\n---\n\n');

      const modeInstruction =
        mode === 'compare'
          ? 'Compare and contrast the selected documents explicitly, citing each filename and section.'
          : mode === 'spreadsheet'
          ? 'Analyze numerical trends, totals, anomalies, and column structure from the tabular excerpts.'
          : mode === 'report'
          ? 'Produce a structured executive report with Overview, Key Evidence, Data Points, and Actionable Recommendations.'
          : 'Answer the question strictly using the retrieved document excerpts. If the information is not present in the excerpts, state that clearly.';

      const result = await generateStructuredText({
        modelId: 'gemini-4-hydrogen',
        thinkingMode: 'high',
        systemInstruction: `You are NOVA Document Intelligence. ${modeInstruction} Always cite sources inline like [Source 1]. Never fabricate facts not present in the excerpts.`,
        prompt: `Retrieved Excerpts:\n${contextBlock}\n\nUser Request: ${queryText}`,
      });

      recordAiUsage({
        userId,
        modelId: 'gemini-4-hydrogen',
        provider: 'gemini',
        endpoint: 'document-rag',
        tokensUsed: result.tokensUsed,
        latencyMs: result.latencyMs,
      });

      res.json({
        answer: result.text,
        sources: chunks.map((c, i) => ({
          index: i + 1,
          filename: c.filename,
          sectionLabel: c.sectionLabel,
          excerpt: c.content.slice(0, 260) + (c.content.length > 260 ? '...' : ''),
          score: Number(c.score.toFixed(3)),
        })),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Document query failed';
      logSystemError('/api/documents/query', msg, req.user!.id);
      res.status(500).json({ error: msg });
    }
  });

  app.delete('/api/documents/:id', requireAuth, (req: AuthenticatedRequest, res) => {
    const result = db
      .prepare('DELETE FROM uploaded_files WHERE id = ? AND user_id = ?')
      .run(req.params.id, req.user!.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Document not found.' });
    res.json({ success: true });
  });

  // ==========================================
  // LONG-TERM PRIVATE MEMORY
  // ==========================================
  app.get('/api/memories', requireAuth, (req: AuthenticatedRequest, res) => {
    const rows = db
      .prepare('SELECT id, user_id, category, content, created_at, updated_at FROM user_memories WHERE user_id = ? ORDER BY updated_at DESC')
      .all(req.user!.id) as Array<Record<string, unknown>>;
    res.json({
      memories: rows.map((r) => ({
        id: r.id,
        userId: r.user_id,
        category: r.category,
        content: r.content,
        createdAt: r.created_at,
        updatedAt: r.updated_at,
      })),
    });
  });

  app.post('/api/memories', requireAuth, async (req: AuthenticatedRequest, res) => {
    const { category, content } = req.body as { category?: string; content?: string };
    const cleanContent = (content || '').trim();
    if (!cleanContent || cleanContent.length < 4) {
      return res.status(400).json({ error: 'Memory content must be at least 4 characters.' });
    }
    const validCat = ['preference', 'project', 'fact', 'instruction'].includes(category || '') ? category! : 'fact';
    const embedding = await generateEmbedding(cleanContent);
    const id = newId();
    db.prepare(
      'INSERT INTO user_memories (id, user_id, category, content, embedding_json) VALUES (?, ?, ?, ?, ?)'
    ).run(id, req.user!.id, validCat, cleanContent, embedding ? JSON.stringify(embedding) : null);

    const row = db.prepare('SELECT id, user_id, category, content, created_at, updated_at FROM user_memories WHERE id = ?').get(id) as Record<string, unknown>;
    res.status(201).json({
      memory: {
        id: row.id,
        userId: row.user_id,
        category: row.category,
        content: row.content,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
      },
    });
  });

  app.patch('/api/memories/:id', requireAuth, async (req: AuthenticatedRequest, res) => {
    const { content, category } = req.body as { content?: string; category?: string };
    const existing = db
      .prepare('SELECT * FROM user_memories WHERE id = ? AND user_id = ?')
      .get(req.params.id, req.user!.id) as Record<string, unknown> | undefined;
    if (!existing) return res.status(404).json({ error: 'Memory not found.' });

    const nextContent = (content ?? (existing.content as string)).trim();
    const nextCat = ['preference', 'project', 'fact', 'instruction'].includes(category || '')
      ? category!
      : (existing.category as string);
    const embedding = await generateEmbedding(nextContent);

    db.prepare(
      "UPDATE user_memories SET category = ?, content = ?, embedding_json = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?"
    ).run(
      nextCat,
      nextContent,
      embedding ? JSON.stringify(embedding) : (existing.embedding_json as string | null),
      req.params.id,
      req.user!.id
    );

    res.json({ success: true });
  });

  app.delete('/api/memories/:id', requireAuth, (req: AuthenticatedRequest, res) => {
    const result = db
      .prepare('DELETE FROM user_memories WHERE id = ? AND user_id = ?')
      .run(req.params.id, req.user!.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Memory not found.' });
    res.json({ success: true });
  });

  // ==========================================
  // AI TOOLS MARKETPLACE (15 WORKFLOWS)
  // ==========================================
  app.get('/api/tools', (_req, res) => {
    res.json({
      tools: AI_TOOLS_CATALOG.map(({ buildPrompt: _bp, ...rest }) => rest),
    });
  });

  app.post('/api/tools/:toolId/run', requireAuth, async (req: AuthenticatedRequest, res) => {
    try {
      const tool = AI_TOOLS_CATALOG.find((t) => t.id === req.params.toolId);
      if (!tool) return res.status(404).json({ error: 'Tool not found.' });

      const inputs = (req.body.inputs || {}) as Record<string, string>;
      for (const f of tool.fields) {
        if (f.required && !String(inputs[f.key] || '').trim()) {
          return res.status(400).json({ error: `${f.label} is required.` });
        }
      }

      const { system, prompt } = tool.buildPrompt(inputs);
      const result = await generateStructuredText({
        modelId: 'gemini-4-hydrogen',
        systemInstruction: system,
        prompt,
        useSearch: Boolean(tool.useSearch),
        thinkingMode: tool.thinkingMode || 'medium',
      });

      recordAiUsage({
        userId: req.user!.id,
        modelId: 'gemini-4-hydrogen',
        provider: 'gemini',
        endpoint: `tool:${tool.id}`,
        tokensUsed: result.tokensUsed,
        latencyMs: result.latencyMs,
      });

      res.json({
        output: result.text,
        citations: result.citations,
        tokensUsed: result.tokensUsed,
        latencyMs: result.latencyMs,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Tool execution failed';
      logSystemError('/api/tools/run', msg, req.user!.id);
      res.status(500).json({ error: msg });
    }
  });

  // ==========================================
  // CODING WORKSPACE
  // ==========================================
  app.get('/api/workspace/projects', requireAuth, (req: AuthenticatedRequest, res) => {
    const rows = db
      .prepare('SELECT * FROM coding_projects WHERE user_id = ? ORDER BY updated_at DESC')
      .all(req.user!.id) as Array<Record<string, unknown>>;

    res.json({
      projects: rows.map((r) => ({
        id: r.id,
        userId: r.user_id,
        name: r.name,
        description: r.description,
        files: JSON.parse((r.files_json as string) || '[]'),
        createdAt: r.created_at,
        updatedAt: r.updated_at,
      })),
    });
  });

  app.post('/api/workspace/projects', requireAuth, (req: AuthenticatedRequest, res) => {
    const { name, description, files } = req.body as {
      name?: string;
      description?: string;
      files?: Array<{ path: string; language: string; content: string }>;
    };
    const id = newId();
    const defaultFiles =
      files && files.length > 0
        ? files
        : [
            {
              path: 'index.html',
              language: 'html',
              content: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NOVA Sandbox</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #09090F; color: #fff; display: grid; place-items: center; min-height: 100vh; }
    .card { padding: 2rem; border-radius: 16px; border: 1px solid rgba(139,92,246,0.35); background: #151520; max-width: 460px; text-align: center; }
    button { margin-top: 1rem; padding: 0.6rem 1.25rem; border-radius: 8px; border: none; background: #8B5CF6; color: #fff; cursor: pointer; font-weight: 600; }
  </style>
</head>
<body>
  <div class="card">
    <h2>NOVA Sandbox Preview</h2>
    <p id="status">Interactive browser sandbox ready.</p>
    <button onclick="document.getElementById('status').textContent = 'Executed safely inside sandboxed iframe at ' + new Date().toLocaleTimeString()">Run Action</button>
  </div>
</body>
</html>`,
            },
          ];

    db.prepare('INSERT INTO coding_projects (id, user_id, name, description, files_json) VALUES (?, ?, ?, ?, ?)').run(
      id,
      req.user!.id,
      (name || 'Untitled Project').trim(),
      (description || '').trim(),
      JSON.stringify(defaultFiles)
    );

    const row = db.prepare('SELECT * FROM coding_projects WHERE id = ?').get(id) as Record<string, unknown>;
    res.status(201).json({
      project: {
        id: row.id,
        userId: row.user_id,
        name: row.name,
        description: row.description,
        files: JSON.parse(row.files_json as string),
        createdAt: row.created_at,
        updatedAt: row.updated_at,
      },
    });
  });

  app.patch('/api/workspace/projects/:id', requireAuth, (req: AuthenticatedRequest, res) => {
    const existing = db
      .prepare('SELECT * FROM coding_projects WHERE id = ? AND user_id = ?')
      .get(req.params.id, req.user!.id) as Record<string, unknown> | undefined;
    if (!existing) return res.status(404).json({ error: 'Project not found.' });

    const { name, description, files } = req.body as {
      name?: string;
      description?: string;
      files?: Array<{ path: string; language: string; content: string }>;
    };

    db.prepare(
      "UPDATE coding_projects SET name = ?, description = ?, files_json = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?"
    ).run(
      name ?? (existing.name as string),
      description ?? (existing.description as string),
      files ? JSON.stringify(files) : (existing.files_json as string),
      req.params.id,
      req.user!.id
    );
    res.json({ success: true });
  });

  app.delete('/api/workspace/projects/:id', requireAuth, (req: AuthenticatedRequest, res) => {
    const result = db
      .prepare('DELETE FROM coding_projects WHERE id = ? AND user_id = ?')
      .run(req.params.id, req.user!.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Project not found.' });
    res.json({ success: true });
  });

  app.post('/api/workspace/assist', requireAuth, async (req: AuthenticatedRequest, res) => {
    try {
      const { action, language, code, instruction } = req.body as {
        action: 'generate' | 'explain' | 'debug' | 'optimize';
        language: string;
        code: string;
        instruction: string;
      };
      const prompts: Record<string, string> = {
        generate: `Generate clean, complete ${language} code for: ${instruction}\n\nExisting file context:\n\`\`\`${language}\n${code}\n\`\`\`\nReturn the updated complete code inside a fenced \`\`\`${language} block followed by brief notes.`,
        explain: `Explain this ${language} code clearly, covering architecture, data flow, and edge cases:\n\n\`\`\`${language}\n${code}\n\`\`\`\nAdditional focus: ${instruction || 'General walkthrough'}`,
        debug: `Find bugs, race conditions, security flaws, and type errors in this ${language} code. Provide the fixed complete code inside a \`\`\`${language} block.\n\nIssue description: ${instruction || 'General audit'}\n\n\`\`\`${language}\n${code}\n\`\`\``,
        optimize: `Refactor and optimize this ${language} code for clarity, maintainability, and performance. Provide the refactored code inside a \`\`\`${language} block.\n\nGoals: ${instruction || 'Clean refactoring'}\n\n\`\`\`${language}\n${code}\n\`\`\``,
      };

      const result = await generateStructuredText({
        modelId: 'gemini-4-hydrogen',
        thinkingMode: 'high',
        systemInstruction: 'You are a principal software architect inside NOVA Coding Workspace. Never run untrusted code on the server.',
        prompt: prompts[action] || prompts.generate,
      });

      recordAiUsage({
        userId: req.user!.id,
        modelId: 'gemini-4-hydrogen',
        provider: 'gemini',
        endpoint: `workspace:${action}`,
        tokensUsed: result.tokensUsed,
        latencyMs: result.latencyMs,
      });

      res.json({ output: result.text, tokensUsed: result.tokensUsed, latencyMs: result.latencyMs });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Workspace assistant failed';
      logSystemError('/api/workspace/assist', msg, req.user!.id);
      res.status(500).json({ error: msg });
    }
  });

  // ==========================================
  // USER DASHBOARD SUMMARY
  // ==========================================
  app.get('/api/dashboard/summary', requireAuth, (req: AuthenticatedRequest, res) => {
    const user = req.user!;
    const convCount = (db.prepare('SELECT COUNT(*) as c FROM conversations WHERE user_id = ?').get(user.id) as { c: number }).c;
    const msgCount = (db.prepare('SELECT COUNT(*) as c FROM messages WHERE user_id = ?').get(user.id) as { c: number }).c;
    const imgCount = (db.prepare('SELECT COUNT(*) as c FROM generated_images WHERE user_id = ?').get(user.id) as { c: number }).c;
    const docCount = (db.prepare('SELECT COUNT(*) as c FROM uploaded_files WHERE user_id = ?').get(user.id) as { c: number }).c;
    const projCount = (db.prepare('SELECT COUNT(*) as c FROM coding_projects WHERE user_id = ?').get(user.id) as { c: number }).c;
    const memCount = (db.prepare('SELECT COUNT(*) as c FROM user_memories WHERE user_id = ?').get(user.id) as { c: number }).c;

    const usageAgg = db
      .prepare('SELECT COALESCE(SUM(tokens_used), 0) as tokens, COALESCE(AVG(latency_ms), 0) as latency FROM ai_usage WHERE user_id = ?')
      .get(user.id) as { tokens: number; latency: number };

    const modelBreakdown = db
      .prepare(
        `SELECT model_id as modelId, COUNT(*) as requests, COALESCE(SUM(tokens_used), 0) as tokens, CAST(COALESCE(AVG(latency_ms), 0) AS INTEGER) as avgLatencyMs
         FROM ai_usage WHERE user_id = ? GROUP BY model_id ORDER BY requests DESC`
      )
      .all(user.id);

    const recentUsage = db
      .prepare('SELECT id, endpoint, model_id, tokens_used, created_at FROM ai_usage WHERE user_id = ? ORDER BY created_at DESC LIMIT 10')
      .all(user.id) as Array<{ id: string; endpoint: string; model_id: string; tokens_used: number; created_at: string }>;

    const limits = getPlanLimits(user.plan);

    res.json({
      summary: {
        totalConversations: convCount,
        totalMessages: msgCount,
        totalTokens: usageAgg.tokens,
        avgLatencyMs: Math.round(usageAgg.latency),
        totalImages: imgCount,
        totalDocuments: docCount,
        totalProjects: projCount,
        totalMemories: memCount,
        dailyQuotaUsed: getDailyMessageUsage(user.id),
        dailyQuotaLimit: limits.dailyMessages,
        imageQuotaUsed: getMonthlyImageUsage(user.id),
        imageQuotaLimit: limits.monthlyImages,
        modelBreakdown,
        recentActivity: recentUsage.map((u) => ({
          id: u.id,
          action: u.endpoint,
          detail: `${u.model_id} · ${u.tokens_used} tokens`,
          createdAt: u.created_at,
        })),
      },
    });
  });

  // ==========================================
  // PROTECTED ADMIN PANEL
  // ==========================================
  app.get('/api/admin/overview', requireAdmin, (_req: AuthenticatedRequest, res) => {
    const users = db
      .prepare('SELECT id, email, name, role, plan, created_at FROM users ORDER BY created_at DESC')
      .all();

    const usageTotals = db
      .prepare(
        'SELECT COUNT(*) as totalRequests, COALESCE(SUM(tokens_used), 0) as totalTokens, CAST(COALESCE(AVG(latency_ms), 0) AS INTEGER) as avgLatencyMs FROM ai_usage'
      )
      .get();

    const errorLogs = db
      .prepare('SELECT id, user_id, route, error_message, severity, created_at FROM error_logs ORDER BY created_at DESC LIMIT 40')
      .all();

    const auditLogs = db
      .prepare(
        `SELECT a.id, u.email as adminEmail, a.action, a.target, a.created_at
         FROM admin_audit_logs a JOIN users u ON u.id = a.admin_user_id
         ORDER BY a.created_at DESC LIMIT 30`
      )
      .all();

    const dbStats = fs.existsSync(process.env.NOVA_DB_PATH || path.resolve(process.cwd(), 'data', 'nova.db'))
      ? fs.statSync(process.env.NOVA_DB_PATH || path.resolve(process.cwd(), 'data', 'nova.db')).size
      : 0;

    res.json({
      users,
      providers: getProviderStatuses(),
      usageTotals,
      errorLogs,
      auditLogs,
      settings: getPlatformSettings(),
      storageBytes: dbStats,
    });
  });

  app.patch('/api/admin/users/:id', requireAdmin, (req: AuthenticatedRequest, res) => {
    const { role, plan } = req.body as { role?: UserRole; plan?: PlanTier };
    const target = db.prepare('SELECT id, email, role, plan FROM users WHERE id = ?').get(req.params.id) as
      | { id: string; email: string; role: UserRole; plan: PlanTier }
      | undefined;
    if (!target) return res.status(404).json({ error: 'User not found.' });

    const nextRole = role === 'admin' || role === 'user' ? role : target.role;
    const nextPlan = plan === 'free' || plan === 'pro' || plan === 'premium' ? plan : target.plan;

    db.prepare("UPDATE users SET role = ?, plan = ?, updated_at = datetime('now') WHERE id = ?").run(
      nextRole,
      nextPlan,
      target.id
    );
    db.prepare('INSERT INTO admin_audit_logs (id, admin_user_id, action, target) VALUES (?, ?, ?, ?)').run(
      newId(),
      req.user!.id,
      `Updated role=${nextRole}, plan=${nextPlan}`,
      target.email
    );
    res.json({ success: true });
  });

  app.patch('/api/admin/settings', requireAdmin, (req: AuthenticatedRequest, res) => {
    const current = getPlatformSettings();
    const updated = { ...current, ...req.body };
    db.prepare(
      "INSERT INTO system_settings (key, value_json, updated_at) VALUES ('platform', ?, datetime('now')) ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = datetime('now')"
    ).run(JSON.stringify(updated));
    db.prepare('INSERT INTO admin_audit_logs (id, admin_user_id, action, target) VALUES (?, ?, ?, ?)').run(
      newId(),
      req.user!.id,
      'Updated platform settings',
      'system_settings'
    );
    res.json({ settings: updated });
  });

  // ==========================================
  // VITE DEV MIDDLEWARE OR PRODUCTION STATIC
  // ==========================================
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const httpServer = http.createServer(app);
  const liveWss = new WebSocketServer({ noServer: true });

  const parseWsToken = (req: http.IncomingMessage, url: URL): string | null => {
    const qToken = url.searchParams.get('token');
    if (qToken) return qToken;
    const cookieHeader = req.headers.cookie || '';
    for (const part of cookieHeader.split(';')) {
      const [k, ...rest] = part.trim().split('=');
      if (k === 'nova_session') {
        return decodeURIComponent(rest.join('='));
      }
    }
    return null;
  };

  httpServer.on('upgrade', (req, socket, head) => {
    try {
      const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
      if (url.pathname !== '/api/live-voice') return;

      const token = parseWsToken(req, url);
      const decoded = token ? verifyAuthToken(token) : null;
      const user = decoded ? getUserById(decoded.sub) : null;
      if (!user) {
        socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
        socket.destroy();
        return;
      }

      liveWss.handleUpgrade(req, socket, head, (ws) => {
        liveWss.emit('connection', ws, req, user, url);
      });
    } catch {
      socket.destroy();
    }
  });

  liveWss.on(
    'connection',
    async (clientWs: WebSocket, _req: http.IncomingMessage, user: NonNullable<ReturnType<typeof getUserById>>, url: URL) => {
      const ai = getGeminiBetaClient();
      if (!ai) {
        clientWs.send(JSON.stringify({ type: 'error', message: 'GEMINI_API_KEY is not configured on the server.' }));
        clientWs.close();
        return;
      }

      const prefs = getUserPreferences(user.id);
      const allowedVoices = ['Zephyr', 'Kore', 'Puck', 'Charon', 'Fenrir'];
      const reqVoice = url.searchParams.get('voice') || prefs.voiceName || 'Zephyr';
      const voiceName = allowedVoices.includes(reqVoice) ? reqVoice : 'Zephyr';
      const language = url.searchParams.get('language') || prefs.language || 'auto';
      const learningMode = url.searchParams.get('learningMode') === '1' || url.searchParams.get('learningMode') === 'true';

      const memories = prefs.memoryEnabled ? await retrieveRelevantMemories(user.id, 'voice call', 4) : [];
      const systemInstruction = buildLiveVoiceSystemInstruction({
        language,
        learningMode,
        voiceName,
        memories: memories.map((m) => `[${m.category}] ${m.content}`),
      });

      const sessionStart = Date.now();
      let liveSession: Awaited<ReturnType<typeof ai.live.connect>> | null = null;
      let turnCount = 0;

      const safeSend = (payload: Record<string, unknown>) => {
        if (clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify(payload));
        }
      };

      try {
        liveSession = await ai.live.connect({
          model: 'gemini-3.8-live',
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName } },
            },
            inputAudioTranscription: {},
            outputAudioTranscription: {},
            systemInstruction,
          },
          callbacks: {
            onopen: () => {
              safeSend({ type: 'ready', voiceName, language, learningMode });
            },
            onmessage: (message: LiveServerMessage) => {
              const content = message.serverContent;
              if (content?.inputTranscription?.text) {
                safeSend({ type: 'input_transcript', text: content.inputTranscription.text });
              }
              if (content?.outputTranscription?.text) {
                safeSend({ type: 'output_transcript', text: content.outputTranscription.text });
              }
              for (const part of content?.modelTurn?.parts ?? []) {
                if (part.inlineData?.data) {
                  safeSend({
                    type: 'audio',
                    audio: part.inlineData.data,
                    mimeType: part.inlineData.mimeType || 'audio/pcm;rate=24000',
                  });
                }
              }
              if (content?.interrupted) {
                safeSend({ type: 'interrupted' });
              }
              if (content?.turnComplete) {
                turnCount += 1;
                safeSend({ type: 'turn_complete' });
              }
            },
            onerror: (err: unknown) => {
              const msg = err instanceof Error ? err.message : 'Live voice session error';
              logSystemError('/api/live-voice', msg, user.id);
              safeSend({ type: 'error', message: msg });
            },
            onclose: () => {
              safeSend({ type: 'closed' });
            },
          },
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Unable to connect to live voice engine.';
        logSystemError('/api/live-voice', msg, user.id);
        safeSend({ type: 'error', message: msg });
        clientWs.close();
        return;
      }

      clientWs.on('message', (rawData) => {
        if (!liveSession) return;
        try {
          const msg = JSON.parse(rawData.toString()) as { type?: string; audio?: string; text?: string };
          if (msg.type === 'audio' && typeof msg.audio === 'string' && msg.audio.length > 0) {
            liveSession.sendRealtimeInput({
              audio: { data: msg.audio, mimeType: 'audio/pcm;rate=16000' },
            });
          } else if (msg.type === 'text' && typeof msg.text === 'string' && msg.text.trim()) {
            liveSession.sendRealtimeInput({ text: msg.text.trim() });
          }
        } catch {
          // ignore malformed client frames
        }
      });

      clientWs.on('close', () => {
        try {
          liveSession?.close();
        } catch {
          // ignore close errors
        }
        recordAiUsage({
          userId: user.id,
          modelId: 'gemini-3.8-live',
          provider: 'gemini',
          endpoint: 'live-voice',
          tokensUsed: Math.max(turnCount * 120, 0),
          latencyMs: Date.now() - sessionStart,
          status: 'ok',
        });
      });
    }
  );

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`NOVA AI server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});
