import React from 'react';

export const NebulaXLogo: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 280 48"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    aria-label="NEBULA-X Research Logo"
  >
    <text
      fill="url(#logo-gradient)"
      xmlSpace="preserve"
      style={{ whiteSpace: 'pre' }}
      fontFamily="Inter, sans-serif"
      fontSize="42"
      fontWeight="900"
      letterSpacing="0.05em"
    >
      <tspan x="0" y="38">NEBULA</tspan>
    </text>
    <path
      d="M218 5L248 43M248 5L218 43"
      stroke="url(#logo-gradient)"
      strokeWidth="6"
      strokeLinecap="round"
    />
    <defs>
      <linearGradient
        id="logo-gradient"
        x1="0"
        y1="24"
        x2="260"
        y2="24"
        gradientUnits="userSpaceOnUse"
      >
        <stop stopColor="#A78BFA" />
        <stop offset="1" stopColor="#818CF8" />
      </linearGradient>
    </defs>
  </svg>
);


export const ChimeraLogo: React.FC<{ className?: string }> = ({ className }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Chimera Project Logo">
        <path d="M12 2L8 10H16L12 2Z" fill="currentColor" fillOpacity="0.8"/>
        <path d="M22 20L18 12H6L2 20H22Z" fill="currentColor" fillOpacity="0.6"/>
        <path d="M7 11L3 19H11L7 11Z" fill="currentColor" />
        <path d="M17 11L21 19H13L17 11Z" fill="currentColor" />
    </svg>
);

export const HNBLogo: React.FC<{ className?: string }> = ({ className }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="HNB Project Logo">
        <path fillRule="evenodd" clipRule="evenodd" d="M12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2ZM12 4C7.58172 4 4 7.58172 4 12C4 16.4183 7.58172 20 12 20C16.4183 20 20 16.4183 20 12C20 7.58172 16.4183 4 12 4Z" fill="currentColor" fillOpacity="0.4"/>
        <path d="M12 8C9.79086 8 8 9.79086 8 12C8 14.2091 9.79086 16 12 16C14.2091 16 16 14.2091 16 12C16 9.79086 14.2091 8 12 8Z" fill="currentColor"/>
    </svg>
);

export const NebulaLogo: React.FC<{ className?: string }> = ({ className }) => (
     <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Nebula Project Logo">
        <circle cx="6" cy="7" r="2" fill="currentColor" fillOpacity="0.8"/>
        <circle cx="12" cy="12" r="3" fill="currentColor" />
        <circle cx="18" cy="17" r="2" fill="currentColor" fillOpacity="0.8"/>
        <path d="M6 7C9.33333 8.33333 10.5 10.5 12 12" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.5" strokeLinecap="round"/>
        <path d="M12 12C13.5 13.5 15.6667 15.6667 18 17" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
);
