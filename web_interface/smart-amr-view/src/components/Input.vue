<template>
  <div class="input-wrapper">
    <div class="input_title">
      <h3>
        {{ title }}
      </h3>

      Please submit sequence
    </div>
      <div class="input_field-wrapper">
        <input ref="input" class="input_field" :placeholder="text" @input="input_text">
        <div class='error' v-if="getError.length > 0">
          {{ getError }}
        </div>
      </div>
      <div class="input_fasta-text">
        <b>or</b> select local fasta file
      </div>
      <div class="input_button-wrapper">
        <input style="opacity: 0;position:absolute" type="file" @change="loadTextFromFile" />
        <button class="input_button">
            Explore 
        </button>
        <div v-if="selected.length > 0" class="input_button-selected">
          {{ selected }}
        </div>
        <div v-else>
          No file selected
        </div>
      </div>
      <div class="input_select-text">
        <b>or</b> select in the database species
      </div>
      <div class="input_select">
        <SelectVue :options="selection" @selected="emitSelected" />
      </div>
    </div>
</template>
<script>
import SelectVue from './Select.vue'
export default {
    name: 'Input',
    components: {
      SelectVue
    },
    data() {
      return {
        current_text: '',
        current_name: '',
        selected: '',
        error: '',
        fileContent: '',
        aminoAcids: ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
      }
    },
    props: {
      text:{
        type:String,
      },
      title:{
        type:String,
      },
      selection:{
        type:Array
      }
    },
    watch: {
      getFileContent(newValue, oldValue) {
        this.loadedContent()
      }
    },
    computed: {
      getError() {
        return this.error
      },
      getFileContent() {
        return this.fileContent
      }
    },
    methods: {
      input_text(event) {
        if (this.rules(event.target.value)) {
          this.error = ''
          this.current_text = event.target.value
          this.emitText(this.current_text)
        } else {
          this.error = event.target.value + ' is not a protein !'
          this.emitError()
        }
      },
      loadedContent() {
        var name, sequence = this.parseFasta(this.getFileContent)
          if (sequence !== '') {
            this.current_text = sequence
            this.current_name = name
            this.emitText(this.current_text)
            this.emitName(this.current_name)
            console.log(name, sequence)
            this.$refs.input.value = name
          } else {
            this.selected = this.selected + ' could not be parsed'
          }
      },
      parseFasta(fasta) {
        const lines = fasta.split("\n");
        const ref = lines[0]
        if (ref[0] === '>') {
          var name = ref.substring(1, ref.length)
          var seq = ''
          let i = 1
          while (i < lines.length) {
            let line = lines[i]
            if (line[0] === '>') {
              break
            }
            seq = seq + line
            i = i + 1
          }
          return name, seq.replace('\n', '')
        } else {
          return ''
        }
      },
      async loadTextFromFile(ev) {
        const envfile = ev.target.files[0]
        console.log(envfile)
        if (envfile.name.includes(".fasta") || envfile.name.includes(".fa")) {
          const reader = new FileReader();
          reader.onload = (res) => {
            this.fileContent = res.target.result;
          };
          reader.readAsText(envfile);
          this.selected = envfile.name
        } else {
          this.selected = file.name + ' is not a fasta file'
        }
      },
      emitError() {
        this.$emit('error')
      },
      emitText(item) {
         this.$emit('update-text', item)
      },
      emitName(item) {
         this.$emit('update-name', item)
      },
      emitSelected(item) {
        this.$emit('selected', item)
      },
      rules(sequence) {
        const re = new RegExp("^["+ this.aminoAcids.join("")+"]*$")
        return sequence.length === 0 || sequence.match(re)
      }
    }
  }

</script>


<style scoped>
.input_title {
  padding: 5px;
  margin: 5px;
}
.error {
  padding:5px;
  font-size:10;
  color:red;
}
.input-wrapper {
  display: block;
  min-height: 2rem;
  width: 100%;
}
.input_field-wrapper {
  width: 100%;
  padding: 5px
}
.input_fasta-text {
  padding: 5px
}

.input_select-text {
  padding: 5px
}

.input_select {
  border: none;
  padding: 5px;
  margin-right: 5px
}

.input_field {
  width: 100%;
  min-height: 2rem;
}
.input_interaction {
  display: flex;
  justify-content: space-between;
}
.input_button {
  border: none;
  padding: 5px;
  margin-right: 5px
}
.input_button-wrapper {
  display: flex;
  margin: 5px;
  justify-content: left;
}

</style>
