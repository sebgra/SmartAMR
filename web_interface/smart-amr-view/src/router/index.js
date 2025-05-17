import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'onevsone',
      component: () => import('../views/MainView.vue'),
      props:{
        bacteria: true,
        phage: true
      }
    },
    {
      path: '/one_vs_one',
      name: 'one_vs_one',
      component: () => import('../views/MainView.vue'),
      props:{
        bacteria: true,
        phage: true
      }
    },
    {
      path: '/bacteria_vs_all',
      name: 'bacteria_vs_all',
      component: () => import('../views/MainView.vue'),
      props:{
        bacteria: true,
        phage: false
      }
    },
    {
      path: '/phage_vs_all',
      name: 'phage_vs_all',
      component: () => import('../views/MainView.vue'),
      props:{
        bacteria: false,
        phage: true
      }
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue')
    }
  ]
})

export default router
