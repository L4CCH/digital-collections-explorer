import viewers from 'virtual:dce-viewers';

export const viewer = (name) => viewers[name] || null;
