const CLERK_SCRIPT_URL = 'https://cdn.jsdelivr.net/npm/@clerk/clerk-js@latest/dist/clerk.browser.js';
let clerkLoadPromise = null;

function getPublishableKey() {
  return import.meta.env.VITE_CLERK_PUBLISHABLE_KEY || '';
}

function getStoredClerkInstance() {
  return window.__ecosightClerk || null;
}

function mapClerkUser(user) {
  if (!user) return null;
  const fullName = [user.firstName, user.lastName].filter(Boolean).join(' ') || user.username || user.primaryEmailAddress?.emailAddress || 'Clerk User';
  return {
    email: user.primaryEmailAddress?.emailAddress || '',
    name: fullName,
    role: 'AUTHENTICATED',
    authProvider: 'clerk',
    id: user.id,
  };
}

export function isClerkEnabled() {
  return Boolean(getPublishableKey());
}

export async function initClerk() {
  if (!isClerkEnabled()) return null;
  if (getStoredClerkInstance()) return getStoredClerkInstance();
  if (!clerkLoadPromise) {
    clerkLoadPromise = new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${CLERK_SCRIPT_URL}"]`);
      if (existing) {
        existing.addEventListener('load', resolve, { once: true });
        existing.addEventListener('error', reject, { once: true });
        return;
      }
      const script = document.createElement('script');
      script.src = CLERK_SCRIPT_URL;
      script.async = true;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    }).then(async () => {
      if (!window.Clerk) {
        throw new Error('Clerk script failed to expose window.Clerk');
      }
      const clerk = new window.Clerk(getPublishableKey());
      await clerk.load();
      window.__ecosightClerk = clerk;
      return clerk;
    });
  }
  return clerkLoadPromise;
}

export async function openClerkAuth(mode = 'login') {
  const clerk = await initClerk();
  if (!clerk) return;
  if (mode === 'signup') {
    clerk.openSignUp();
    return;
  }
  clerk.openSignIn();
}

export async function signOutClerk() {
  const clerk = await initClerk();
  if (!clerk) return;
  await clerk.signOut();
}

export async function getClerkUser() {
  const clerk = await initClerk();
  return mapClerkUser(clerk?.user || null);
}

export async function attachClerkListener(callback) {
  const clerk = await initClerk();
  if (!clerk || !clerk.addListener) return () => {};
  return clerk.addListener(({ user }) => {
    callback(mapClerkUser(user || null));
  });
}
