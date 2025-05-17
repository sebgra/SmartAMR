<template>
  <div class="select-wrapper">
    <select class="select" id="dropdown" v-model="selectedOption" @change="emitSelection">
      <option disabled value="">Select in database</option>
      <option v-for="option in options" :key="option" :value="option">
        {{ option }}
      </option>
    </select>  
    </div>
</template>

<script>
export default {
  name:'Select',
  props: {
    options:{
        type:Array,
        required: true
    }
  },
  data () {
      return {
        selectedOption: '',
      }
    },
  methods: {
    emitSelection() {
      this.$emit('selected', this.selectedOption)
    },

  }
}
</script>
<style scoped>
.select-wrapper {
  display: flex;
  min-height: 1rem;
  width: 50%;
  justify-content: center;
}

.select {
    width:inherit;
}
</style>