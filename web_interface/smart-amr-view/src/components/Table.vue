<template>
  <div class="table">
    <div class="table_header">
      <div class="table_element"><b class="weight">Bacteria</b></div>
      <div class="table_element"><b class="weight">Phage</b></div>
      <div class="table_element"><b class="weight">Phage efficiency</b></div>
    </div>
    <div :key="line" class="table_line" v-for="line in datas">
      <div class="table_element">
        {{ line.bacteria }}
      </div>
      <div class="table_element">
        {{ line.phage }}
      </div>
      <div class="table_element">
        {{ line.evaluation }}
      </div>
    </div>
  </div>
</template>
<script>
export default {
  name:'Table',
  props: {
    datas: Object
  },
  data() {
      return {
      }
    },
}
</script>


<style scoped>
.table {
  display: block;
  height:fit-content;
  border: 1px solid lightgrey;
  border-radius: 5px;
  justify-content: left;
  overflow:hidden;
}
.table_header {
  display: flex;
  justify-content: space-between;
  background-color: #408d5a;
  color: white;
  border-radius: 5px 5px 0 0;
}
.weight {
  font-weight: 700;
}
.table_line {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid lightgrey;
  overflow: hidden;
}
.table_line:last-child{
  border-bottom: none
}
.table_line:hover {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid lightgrey;
  background-color: lightgrey
}
.table_element {
  width: 33%;
  border-right: 1px solid lightgrey;
  padding: 5px;
  overflow: inherit;
}
</style>
