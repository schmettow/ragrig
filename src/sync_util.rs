//! Poison-resilient locking helpers.
//!
//! A [`std::sync::Mutex`] / [`std::sync::RwLock`] is *poisoned* when a panic
//! unwinds while a guard is held.  Every subsequent lock attempt then
//! returns `Err(PoisonError)`, so naive `.lock().unwrap()` call sites turn
//! one contained panic — e.g. a user-supplied
//! [`Generator`](crate::agents::Generator) panicking mid-turn — into a
//! permanently dead shared resource.  Worst case: dropping a poisoned
//! [`BruteForceStore`](crate::store::BruteForceStore) during unwind panics
//! inside `Drop` and aborts the whole process.
//!
//! These helpers recover the guard instead.  The data guarded through them
//! (vector-store state, generator wrappers, model caches) cannot be left
//! torn by the code that runs under the lock, so "keep going with the data"
//! is the safe choice for a library whose consumers (GUIs) routinely catch
//! panics at frame boundaries.

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Lock `m`, recovering the guard if the mutex was poisoned.
pub(crate) fn lock_or_recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Unwrap a `Mutex::into_inner()` result, recovering the value on poison.
pub(crate) fn into_inner_or_recover<T>(r: Result<T, std::sync::PoisonError<T>>) -> T {
    r.unwrap_or_else(|e| e.into_inner())
}

/// Read-lock `l`, recovering the guard if the lock was poisoned.
pub(crate) fn read_or_recover<T>(l: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}

/// Write-lock `l`, recovering the guard if the lock was poisoned.
pub(crate) fn write_or_recover<T>(l: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(|e| e.into_inner())
}
