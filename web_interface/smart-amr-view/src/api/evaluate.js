const baseURL = 'api/evaluate/'

export default (axiosInstance) => {
  const getEvaluateData = async (payload) => {
    try {

      const result = await axiosInstance.post(baseURL, payload)
      return result.data
    } catch (error) {
      console.error(error)
      throw error
    }
  }

  return {
    getEvaluateData
  }
}