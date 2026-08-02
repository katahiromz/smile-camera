// TopSnackbarWeb.tsx --- Web版TopSnackbarコンポーネント
// Author: katahiromz
// License: MIT

import React, { useEffect, useRef, useState } from 'react';

export interface TopSnackbarWebProps {
  message: string;
  actionLabel?: string;
  onAction?: () => void;
  onClose?: () => void;
  durationMs?: number;
}

const SWIPE_THRESHOLD = 50;
const DISMISS_ANIMATION_MS = 250;

const TopSnackbarWeb: React.FC<TopSnackbarWebProps> = ({
  message,
  actionLabel,
  onAction,
  onClose,
  durationMs = 3000,
}) => {
  const [dismissing, setDismissing] = useState(false);
  const touchStartX = useRef(0);
  const touchStartY = useRef(0);
  const dismissed = useRef(false);

  const dismiss = () => {
    if (dismissed.current) return;
    dismissed.current = true;
    setDismissing(true);
    setTimeout(() => {
      onClose?.();
    }, DISMISS_ANIMATION_MS);
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      dismiss();
    }, durationMs);
    return () => clearTimeout(timer);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartX.current = e.touches[0].clientX;
    touchStartY.current = e.touches[0].clientY;
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    const dx = e.changedTouches[0].clientX - touchStartX.current;
    const dy = e.changedTouches[0].clientY - touchStartY.current;

    if (Math.abs(dy) > Math.abs(dx)) {
      // 縦スワイプ
      if (dy < -SWIPE_THRESHOLD) {
        dismiss(); // 上スワイプで消す
      }
    } else {
      // 横スワイプ
      if (Math.abs(dx) > SWIPE_THRESHOLD) {
        dismiss(); // 左右スワイプで消す
      }
    }
  };

  const handleActionClick = () => {
    onAction?.();
    dismiss();
  };

  return (
    <div
      className={`top-snackbar-web${dismissing ? ' dismissing' : ''}`}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      role="status"
      aria-live="polite"
    >
      <span className="top-snackbar-web-message">{message}</span>
      <div className="top-snackbar-web-actions">
        {actionLabel && (
          <button className="top-snackbar-web-action-btn" onClick={handleActionClick}>
            {actionLabel}
          </button>
        )}
        <button className="top-snackbar-web-close-btn" onClick={dismiss} aria-label="Close">
          ✕
        </button>
      </div>
    </div>
  );
};

export default TopSnackbarWeb;
