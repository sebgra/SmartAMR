<script setup>
import { RouterLink, RouterView } from 'vue-router'
import Title from './components/Title.vue'
</script>

<template>
  <div class="main">
    <header>
      <div class="wrapper">
        <Title msg="SmartAMR" />

        <nav>
          <RouterLink to="/one_vs_one">Bacteria vs Phage</RouterLink>
          <RouterLink to="/bacteria_vs_all">Bacteria vs All</RouterLink>
          <RouterLink to="/phage_vs_all">Phage vs All</RouterLink>
          <RouterLink to="/about">About</RouterLink>
        </nav>
      </div>
    </header>

    <RouterView />
  </div>
</template>

<style scoped>
header {
  line-height: 1.5;
  max-height: 100vh;
}

nav {
  width: 100%;
  font-size: 12px;
  text-align: center;
  margin-top: 2rem;
}

nav a.router-link-exact-active {
  color: var(--color-text);
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav a {
  display: inline-block;
  padding: 0 1rem;
  border-left: 1px solid var(--color-border);
}

nav a:first-of-type {
  border: 0;
}
</style>
