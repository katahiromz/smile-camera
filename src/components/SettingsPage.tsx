// SettingsPage.tsx --- 設定ページ React コンポーネント
// Author: katahiromz
// License: MIT
import React from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowLeft } from 'lucide-react';
import './SettingsPage.css';

// アプリケーションのベースパスを取得
const BASE_URL = import.meta.env.BASE_URL;

interface SettingsPageProps {
  onBack: () => void;
  isClosing?: boolean;
}

const SettingsPage: React.FC<SettingsPageProps> = ({
  onBack,
  isClosing = false
}) => {
  const { t } = useTranslation();

  return (
    <div className={`settings-page ${isClosing ? 'closing' : ''}`}>
      <div className="settings-header">
        <button 
          className="back-button" 
          onClick={onBack}
          aria-label={t('back')}
        >
          <ArrowLeft size={24} />
        </button>
        <h1>{t('settings_page')}</h1>
      </div>
      <div className="settings-content">
        <div className="setting-version-info">
          <p>
            {t('camera_app_info')}
          </p>
          <p>
            <img src={`${BASE_URL}pwa-120x120.png`} alt="[Logo]" />
          </p>
        </div>
        <hr />
        <div className="setting-item">
          {t('no_settings')}
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
