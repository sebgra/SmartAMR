<template>
  <div class="one_vs_one">
    <div v-if="bacteria" class="input">
      <InputVue
        :text="inputText"
        title="Bacteria sequence"
        :selection="bacteriaList"
        @updateText="updateBacteria"
        @updateName="updateBacteriaName"
        @error="setInputError"
        @selected="selectBacteria"
      />
    </div>
    <div v-if="phage" class="input">
      <InputVue 
        :text="inputText"
        title="Phage sequence"
        :selection="phageList"
        @updateText="updatePhage"
        @updateName="updatePhageName"
        @error="setInputError"
        @selected="selectPhage"
      />
    </div>
    <div class="action">
      <Action title="Launch Phage efficiency evaluation" :activated="Boolean(isValidCombination&!getInputError)" @action="sendRequest"/>
    </div>
    <div class="table-view" v-if="hasDatas">
        <h3 style="margin-bottom:5px">
          Evaluation of phage efficiency
        </h3>

      <TableView :datas="getTableDatas"/>
    </div>
    <div v-else-if="getIsLoading">
      <div class="center">
        <img  src="@/assets/loading.gif" alt="Loading" />
      </div>
      <div class="center">
        <h3>Computing affinity...</h3>
      </div>
    </div>
    <div v-if="getIsError">
      An error occured: {{ error }}
    </div>
    <!--
    <div class="export">
      <vue-json-to-csv
        :json-data="getTableJson"
        csv-title="generated_antibodies"
        @success="val => console.log(val)"
        @error="val => console.error(val)"
      >
          <button class="export_button">
            Export 
          </button>
      </vue-json-to-csv>
    </div>
    -->
  </div>
</template>

<script>
import InputVue from "../components/Input.vue"
import SelectVue from "../components/Select.vue"
import Action from "../components/Action.vue"
import TableView from "../components/Table.vue"
import VueJsonToCsv from 'vue-json-to-csv'
import api from '@/api/api.js'
import DB from '../assets/speciesList'

export default {
    name: 'GenerateView',
    props: {
      bacteria:{
        type:Boolean,
        default: true
      },
      phage:{
        type:Boolean,
        default: true
      },
    },
    data() {
      return {
        bacteriaSequence: '',
        phageSequence:'',
        bacteriaSelected:'',
        phageSelected:'',
        isLoading: false,
        isError:false,
        error:'',
        tableDatas: [],
        tableJson: {},
        inputError: false,
        inputText: "Enter a peptidic sequence",
        validBacteria: false,
        validPhage:false
      }
    },
    components: {
      InputVue,
      SelectVue,
      Action,
      TableView,
      VueJsonToCsv,
    },
    watch:{
      bacteriaSequence(newVal) {
        this.validBacteria = Boolean(newVal.length > 0)
        if (this.validBacteria) {
          this.bacteriaSelected = ''
        }
      },
      bacteriaSelected(newVal) {
        this.validBacteria = Boolean(newVal.length > 0)
        if (this.validBacteria) {
          this.bacteriaSequence = ''
        }
      },
      phageSequence(newVal) {
        this.validPhage = Boolean(newVal.length > 0)
        if (this.validPhage) {
          this.phageSelected = ''
        }
      },
      phageSelected(newVal) {
        this.validPhage = Boolean(newVal.length > 0)
        if (this.validPhage) {
          this.phageSequence = ''
        }
      }
    },
    computed: {
      bacteriaList: function() {
        return DB.bacteriasDB
      },
      phageList: function() {
        return DB.phagesDB
      },
      isValidCombination: function() {
        return Boolean((this.validBacteria|!this.bacteria)&(this.validPhage|!this.phage))
      },
      getTableJson: function() {
        return this.tableJson
      },
      getIsLoading: function() {
        return this.isLoading
      },
      getIsError: function() {
        return this.isError
      },
      hasDatas: function() {
        return this.tableDatas.length > 0
      },
      getTableDatas: function() {
        return this.tableDatas
      },
      getInputError: function() {
        return this.inputError
      }
    },
    mounted() {
      if (this.$store.state.antigenesTable.length > 0) {
        this.tableDatas = this.$store.state.antigenesTable
      }
      if (this.$store.state.sequence.length > 0) {
        this.inputText = this.$store.state.sequence
      }
      if (this.$store.state.json.length > 0) {
        this.dataJson = this.$store.state.json
      }
    },
    methods: {
      selectBacteria(species) {
        this.bacteriaSelected = species
        this.inputError = false
      },
      selectPhage(species) {
        this.phageSelected = species
        this.inputError = false
      },
      updateBacteria(emited) {
        this.bacteriaSequence = emited
        this.inputError = false
      },
      updateBacteriaName(emited) {
        this.bacteriaSelected = emited
      },
      updatePhage(emited) {
        this.phageSequence = emited
        this.inputError = false
      },
      updatePhageName(emited) {
        this.phageSelected = emited
      },
      setInputError() {
        this.inputError = true
      },
      transpose(table) {
        var new_table = []
        const num_results = table[0].length
        for (var i = 0; i < num_results; i++) {
          var col_values =  table.map(x => x[i])
          new_table.push(col_values)
        }
        return new_table
      },
      toJson(datas) {
        const json = []
        for (var line in datas) {
          json.push({'Sequence':line[0], 'Score':line[1]})
        }
        return JSON.stringify(json)
      },
      async sendRequest () {
        try {
          this.isError = false
          this.isLoading = true
          // retrieving sequences if selected species
          var bacteriaName = ""
          var phageName = ""
          if (this.bacteria) {
            if (this.bacteriaSelected.length > 0) {
              bacteriaName = this.bacteriaSelected
            } else {
              bacteriaName = "bacterialSeq"
            }
          }
          if (this.phage) {
            if (this.phageSelected.length > 0) {
              phageName = this.phageSelected
            } else {
              phageName = "phageSeq"
            }
          }
          const payload = {
            "bacteria": {
              "name": bacteriaName,
              "sequence": this.bacteriaSequence
            },
            "phage": {
              "name": phageName, 
              "sequence": this.phageSequence
            }
          }
          const res = await api.evaluate.getEvaluateData(payload)
          console.log(res)
          var evaluations = []
          const keys = Object.keys(res.data)
          keys.forEach(key => {
            evaluations.push(res.data[key])
          })
          this.tableDatas = evaluations
          this.isLoading = false
          this.$store.commit('setResultTable', res)
        } catch (error) {
          this.isLoading = false
          this.error = error
          this.isError = true
        }
      },
    }
  }

</script>

<style>
.one_vs_one {
  display: block;
  align-items: center;
  padding: 5px;
  margin-top: 10px
}

.loader {
  margin:auto;
  display: block;
  justify-content: center;
}

.export {
  margin:auto;
  width: 100%;
  display: flex;
  justify-content: center;
}

.export_button {
  background-color:#409d9e;
  border: none;
  border-radius: 2px;
  color: white;
  padding:5px;
  font-weight: 700;
}
.center {
  margin: auto;
  display: flex;
  justify-content: center;
}

.input {
  width: 100%;
  align-items: center;
  border: 1px solid lightgrey;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px
}

.action {
  width: 100%;
  margin-bottom:10px;
}

.table-view {
  width: 100%;
  padding: 10px;
}
</style>