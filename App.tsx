import React, { Component, useCallback, useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { AppShell } from './components/ui/AppShell';
import { AuthPage } from './pages/AuthPage';
import { ChatPage } from './pages/ChatPage';
import { ImageStudioPage } from './pages/ImageStudioPage';
import { ToolsMarketplacePage } from './pages/ToolsMarketplacePage';
import { CodingWorkspacePage } from './pages/CodingWorkspacePage';
import { DocumentsPage } from './pages/DocumentsPage';
import { MemoryPage } from './pages/MemoryPage';
import { HistoryPage } from './pages/HistoryPage';
import { DashboardPage } from './pages/DashboardPage';
import { SettingsPage } from './pages/SettingsPage';
import { SubscriptionPage } from './pages/SubscriptionPage';
import { AdminPage } from './pages/AdminPage';
import { VoiceCallPage } from './pages/VoiceCallPage';

class AppErrorBoundary extends Component<{ children: React.ReactNode }, { hasError: boolean; errorMessage: string }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, errorMessage: '' };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, errorMessage: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#09090B] text-white flex flex-col items-center justify-center p-6 text-center">
          <h1 className="font-display text-2xl font-bold">Something went wrong</h1>
          <p className="mt-2 text-sm text-zinc-400 max-w-md">{this.state.errorMessage}</p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-6 px-5 py-2.5 rounded-xl bg-white text-zinc-950 text-sm font-medium cursor-pointer"
          >
            Reload NOVA AI
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function NovaRouter() {
  const { user, loading } = useAuth();
  const [path, setPath] = useState<string>(() => window.location.pathname || '/');
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [newChatKey, setNewChatKey] = useState(0);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [pendingSnippet, setPendingSnippet] = useState<{ code: string; language: string } | null>(null);

  const navigate = useCallback((nextPath: string) => {
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, '', nextPath);
    }
    setPath(nextPath);
  }, []);

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname || '/');
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const handleOpenCodeInWorkspace = useCallback(
    (code: string, language: string) => {
      setPendingSnippet({ code, language });
      navigate('/workspace');
    },
    [navigate]
  );

  const handleNewChat = useCallback(() => {
    setActiveConversationId(null);
    setNewChatKey((k) => k + 1);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#09090B] text-white flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-zinc-400" />
      </div>
    );
  }

  if (!user) {
    return <AuthPage initialMode={path === '/register' ? 'register' : 'login'} navigate={navigate} />;
  }

  const renderWorkspaceContent = () => {
    switch (path) {
      case '/voice':
        return <VoiceCallPage />;
      case '/image-studio':
        return <ImageStudioPage />;
      case '/tools':
        return (
          <ToolsMarketplacePage
            onOpenCodeInWorkspace={handleOpenCodeInWorkspace}
            navigate={navigate}
          />
        );
      case '/workspace':
        return (
          <CodingWorkspacePage
            pendingSnippet={pendingSnippet}
            onClearPendingSnippet={() => setPendingSnippet(null)}
          />
        );
      case '/documents':
        return <DocumentsPage />;
      case '/memory':
        return <MemoryPage />;
      case '/history':
        return (
          <HistoryPage
            onSelectConversation={(id) => setActiveConversationId(id)}
            navigate={navigate}
          />
        );
      case '/dashboard':
        return <DashboardPage navigate={navigate} />;
      case '/settings':
        return <SettingsPage />;
      case '/subscription':
        return <SubscriptionPage />;
      case '/admin':
        return <AdminPage />;
      case '/chat':
      case '/':
      default:
        return (
          <ChatPage
            conversationId={activeConversationId}
            newChatKey={newChatKey}
            onConversationCreated={(id) => {
              setActiveConversationId(id);
              setRefreshTrigger((n) => n + 1);
            }}
            onConversationUpdated={() => setRefreshTrigger((n) => n + 1)}
            onOpenCodeInWorkspace={handleOpenCodeInWorkspace}
            navigate={navigate}
          />
        );
    }
  };

  return (
    <AppShell
      currentPath={path}
      navigate={navigate}
      activeConversationId={activeConversationId}
      onSelectConversation={(id) => setActiveConversationId(id)}
      onNewChat={handleNewChat}
      refreshTrigger={refreshTrigger}
    >
      {renderWorkspaceContent()}
    </AppShell>
  );
}

export default function App() {
  return (
    <AppErrorBoundary>
      <AuthProvider>
        <NovaRouter />
      </AuthProvider>
    </AppErrorBoundary>
  );
}
