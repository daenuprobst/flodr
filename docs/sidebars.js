// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  guide: [
    'index',
    'installation',
    'quickstart',
    {
      type: 'category',
      label: 'User guide',
      collapsed: false,
      items: [
        'guide/fitting',
        'guide/inputs',
        'guide/new-points',
        'guide/inverse',
        'guide/density',
        'guide/diagnostics',
        'guide/scikit-learn',
        'guide/performance',
      ],
    },
  ],
  api: ['api/flodr', 'api/viz', 'api/training'],
};

export default sidebars;
