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
        'guide/new-points',
        'guide/inverse',
        'guide/density',
        'guide/audit',
        'guide/visualisation',
        'guide/scikit-learn',
      ],
    },
  ],
  api: ['api/flodr', 'api/viz', 'api/training'],
};

export default sidebars;
