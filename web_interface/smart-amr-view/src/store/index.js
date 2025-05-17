import { createStore } from 'vuex'

export default createStore({
  state: {
    antigenesTable: [],
    sequence: '',
    json: []
  },
  mutations: {
    setResultTable(state, payload) {
      state.resultTable = payload.data
    },
    setJson(state, payload) {
      state.json= payload.data
    },
  }
})
