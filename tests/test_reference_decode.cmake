if(NOT DEFINED PROBE OR NOT DEFINED FFMPEG OR NOT DEFINED WORK)
  message(FATAL_ERROR "PROBE, FFMPEG and WORK are required")
endif()
file(MAKE_DIRECTORY "${WORK}")
# Spaces and shell metacharacters must stay part of the file name.
set(clip "${WORK}/reference & clip.mkv")
set(audio "${WORK}/reference audio.wav")
execute_process(COMMAND "${FFMPEG}" -v error -nostdin -y
  -f lavfi -i "color=c=red:size=4x4:rate=24:duration=2"
  -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=2"
  -c:v ffv1 -c:a pcm_f32le "${clip}"
  RESULT_VARIABLE status ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Could not create video/audio fixture: ${error}")
endif()
execute_process(COMMAND "${FFMPEG}" -v error -nostdin -y -i "${clip}" -vn
  -c:a pcm_f32le "${audio}" RESULT_VARIABLE status ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Could not create standalone audio fixture: ${error}")
endif()
set(delayed "${WORK}/delayed soundtrack.mkv")
execute_process(COMMAND "${FFMPEG}" -v error -nostdin -y
  -f lavfi -i "color=c=red:size=4x4:rate=24:duration=3"
  -itsoffset 1 -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=2"
  -c:v ffv1 -c:a pcm_f32le "${delayed}"
  RESULT_VARIABLE status ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Could not create delayed audio fixture: ${error}")
endif()
execute_process(COMMAND "${PROBE}" "${clip}" "${audio}" "${delayed}"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Reference ingestion failed: ${status}\n${output}\n${error}")
endif()
