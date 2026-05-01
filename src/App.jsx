import { useEffect, useMemo, useRef, useState } from 'react'
import * as XLSX from 'xlsx'
import './App.css'

const STUDENTS_KEY = 'smartface_students_v3'
const ATT_PREFIX = 'smartface_attendance_'
const GROUP_ATT_PREFIX = 'smartface_group_attendance_'
const FACE_THRESHOLD = 0.52
const SCAN_INTERVAL_MS = 1500

const MODEL_URLS = [
  'https://justadudewhohacks.github.io/face-api.js/models',
  'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights',
]

const FACE_API_SOURCES = [
  'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js',
  'https://unpkg.com/face-api.js@0.22.2/dist/face-api.min.js',
]

function readJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

function writeJSON(key, value) {
  localStorage.setItem(key, JSON.stringify(value))
}

function todayKey() {
  return ATT_PREFIX + new Date().toISOString().slice(0, 10)
}

function safeGroupName(groupName) {
  const cleaned = (groupName || '').trim()
  return cleaned || 'Unassigned'
}

function groupStorageKey(groupName) {
  return GROUP_ATT_PREFIX + encodeURIComponent(safeGroupName(groupName))
}

function groupFromStorageKey(key) {
  return decodeURIComponent(key.replace(GROUP_ATT_PREFIX, ''))
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function euclidean(a, b) {
  let sum = 0
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }
  return Math.sqrt(sum)
}

function formatTimestampLabel(ts) {
  const date = new Date(ts)
  if (Number.isNaN(date.getTime())) return String(ts)
  return date.toLocaleString()
}

function App() {
  const [students, setStudents] = useState(() => readJSON(STUDENTS_KEY, []))
  const [attendance, setAttendance] = useState(() => readJSON(todayKey(), []))

  const [studentName, setStudentName] = useState('')
  const [studentGroup, setStudentGroup] = useState('')
  const [studentPhoto, setStudentPhoto] = useState(null)
  const [groupPhoto, setGroupPhoto] = useState(null)
  const [groupScopeName, setGroupScopeName] = useState('')

  const [filterGroup, setFilterGroup] = useState('')
  const [selectedGroupView, setSelectedGroupView] = useState('')

  const [modelsReady, setModelsReady] = useState(false)
  const [modelStatus, setModelStatus] = useState('Loading AI models...')
  const [registerStatus, setRegisterStatus] = useState('')
  const [groupStatus, setGroupStatus] = useState('')
  const [datasetStatus, setDatasetStatus] = useState('')
  const [cameraStatus, setCameraStatus] = useState('Camera is currently offline.')
  const [processedPreview, setProcessedPreview] = useState('')

  const [isCameraOn, setIsCameraOn] = useState(false)
  const [isCapturingRegister, setIsCapturingRegister] = useState(false)

  const videoRef = useRef(null)
  const overlayRef = useRef(null)
  const datasetInputRef = useRef(null)
  const streamRef = useRef(null)
  const scannerTimerRef = useRef(null)

  const filteredAttendance = useMemo(() => {
    const g = filterGroup.trim().toLowerCase()
    if (!g) return attendance
    return attendance.filter((r) => (r.Group || '').toLowerCase() === g)
  }, [attendance, filterGroup])

  const knownGroups = useMemo(() => {
    const groups = new Set()
    students.forEach((s) => {
      if (s.group) groups.add(s.group)
    })
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i)
      if (key && key.startsWith(GROUP_ATT_PREFIX)) {
        groups.add(groupFromStorageKey(key))
      }
    }
    return Array.from(groups).sort((a, b) => a.localeCompare(b))
  }, [students, attendance])

  function buildGroupSheetFromToday(groupName) {
    const group = safeGroupName(groupName)
    const inGroupAttendance = attendance.filter((r) => safeGroupName(r.Group).toLowerCase() === group.toLowerCase())
    const presentSet = new Set(inGroupAttendance.map((r) => r.Name))

    const nameSet = new Set()
    students
      .filter((s) => safeGroupName(s.group).toLowerCase() === group.toLowerCase())
      .forEach((s) => nameSet.add(s.name))
    inGroupAttendance.forEach((r) => nameSet.add(r.Name))

    if (!nameSet.size) {
      return { timestamps: [], rows: [] }
    }

    const ts = new Date().toISOString()
    const rows = Array.from(nameSet)
      .sort((a, b) => a.localeCompare(b))
      .map((name) => ({ Name: name, [ts]: presentSet.has(name) ? 'p' : 'a' }))

    return { timestamps: [ts], rows }
  }

  const selectedGroupSheet = useMemo(() => {
    if (!selectedGroupView) return null
    const key = groupStorageKey(selectedGroupView)
    const stored = readJSON(key, { timestamps: [], rows: [] })
    if ((stored.rows || []).length) return stored

    // Auto-create first matrix view from today's attendance if no sheet exists yet.
    const fallback = buildGroupSheetFromToday(selectedGroupView)
    if ((fallback.rows || []).length) {
      writeJSON(key, fallback)
    }
    return fallback
  }, [selectedGroupView, attendance, students])

  const uniquePresentToday = useMemo(() => {
    return new Set(attendance.map((r) => `${r.Name}|${r.Group || ''}`)).size
  }, [attendance])

  useEffect(() => {
    writeJSON(STUDENTS_KEY, students)
  }, [students])

  useEffect(() => {
    writeJSON(todayKey(), attendance)
  }, [attendance])

  useEffect(() => {
    let mounted = true

    async function bootModels() {
      try {
        await loadFaceApiLibrary()
        await loadModelsFromAnyCdn()
        if (!mounted) return
        setModelsReady(true)
        setModelStatus('AI Ready')
      } catch (err) {
        if (!mounted) return
        setModelStatus('Model load failed')
        setCameraStatus(`Failed to load models: ${err.message}`)
      }
    }

    bootModels()

    return () => {
      mounted = false
      stopCamera()
    }
  }, [])

  function getFaceApi() {
    if (!window.faceapi) {
      throw new Error('face-api.js not loaded.')
    }
    return window.faceapi
  }

  async function loadFaceApiLibrary() {
    if (window.faceapi) return

    for (const src of FACE_API_SOURCES) {
      try {
        // eslint-disable-next-line no-await-in-loop
        await new Promise((resolve, reject) => {
          const script = document.createElement('script')
          script.src = src
          script.async = true
          script.onload = resolve
          script.onerror = reject
          document.head.appendChild(script)
        })
        if (window.faceapi) return
      } catch {
        // try next source
      }
    }

    throw new Error('Unable to load face-api library from CDN.')
  }

  async function loadModelsFromAnyCdn() {
    const faceapi = getFaceApi()
    let lastError = new Error('Model load failed')

    for (const url of MODEL_URLS) {
      try {
        // eslint-disable-next-line no-await-in-loop
        await faceapi.nets.tinyFaceDetector.loadFromUri(url)
        // eslint-disable-next-line no-await-in-loop
        await faceapi.nets.faceLandmark68Net.loadFromUri(url)
        // eslint-disable-next-line no-await-in-loop
        await faceapi.nets.faceRecognitionNet.loadFromUri(url)
        return
      } catch (err) {
        lastError = err
      }
    }

    throw lastError
  }

  async function ensureCamera() {
    if (streamRef.current) return
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    streamRef.current = stream
    videoRef.current.srcObject = stream
    await videoRef.current.play()
    syncOverlaySize()
    setIsCameraOn(true)
    setCameraStatus('Camera started. Ready to scan.')
  }

  function stopCamera() {
    if (scannerTimerRef.current) {
      clearInterval(scannerTimerRef.current)
      scannerTimerRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    clearOverlay()
    setIsCameraOn(false)
    setCameraStatus('Camera is currently offline.')
  }

  function clearOverlay() {
    const canvas = overlayRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }

  function syncOverlaySize() {
    if (!videoRef.current || !overlayRef.current) return
    if (!videoRef.current.videoWidth || !videoRef.current.videoHeight) return

    const canvas = overlayRef.current
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    canvas.style.width = `${videoRef.current.clientWidth}px`
    canvas.style.height = `${videoRef.current.clientHeight}px`
  }

  function matchDescriptor(descriptor, groupFilter = '') {
    const normalizedGroup = groupFilter.trim().toLowerCase()
    let best = { name: 'Unknown', group: '', distance: 999 }

    students.forEach((student) => {
      if (normalizedGroup && (student.group || '').toLowerCase() !== normalizedGroup) return
      ;(student.descriptors || []).forEach((desc) => {
        const d = euclidean(descriptor, desc)
        if (d < best.distance) {
          best = { name: student.name, group: student.group || '', distance: d }
        }
      })
    })

    if (best.distance <= FACE_THRESHOLD) return best
    return { name: 'Unknown', group: '', distance: best.distance }
  }

  function markAttendance(name, group) {
    setAttendance((prev) => {
      const already = prev.some((r) => r.Name === name && (r.Group || '') === (group || ''))
      if (already) return prev
      return [
        ...prev,
        {
          Name: name,
          Group: group || '',
          Time: new Date().toLocaleTimeString(),
          Status: 'Present',
        },
      ]
    })
  }

  function updateGroupMatrix(groupName, presentNames, timestampISO) {
    const group = safeGroupName(groupName)
    const key = groupStorageKey(group)
    const snapshot = readJSON(key, { timestamps: [], rows: [] })
    const ts = timestampISO || new Date().toISOString()

    const allNames = new Set(snapshot.rows.map((r) => r.Name))
    presentNames.forEach((name) => allNames.add(name))
    students
      .filter((s) => safeGroupName(s.group).toLowerCase() === group.toLowerCase())
      .forEach((s) => allNames.add(s.name))

    if (!snapshot.timestamps.includes(ts)) snapshot.timestamps.push(ts)
    snapshot.timestamps.sort()

    const rowMap = {}
    snapshot.rows.forEach((r) => {
      rowMap[r.Name] = { ...r }
    })

    Array.from(allNames)
      .sort((a, b) => a.localeCompare(b))
      .forEach((name) => {
        if (!rowMap[name]) rowMap[name] = { Name: name }
        rowMap[name][ts] = presentNames.includes(name) ? 'p' : 'a'
      })

    const next = {
      timestamps: snapshot.timestamps,
      rows: Object.values(rowMap),
    }
    writeJSON(key, next)
    setAttendance((prev) => [...prev])
  }

  function exportGroupExcel(groupName) {
    const group = safeGroupName(groupName)
    const sheetData = readJSON(groupStorageKey(group), { timestamps: [], rows: [] })
    if (!sheetData.rows.length) {
      setGroupStatus('No attendance data for selected group.')
      return
    }

    const header = ['names', ...sheetData.timestamps.map((ts) => formatTimestampLabel(ts))]
    const body = sheetData.rows.map((row) => [
      row.Name,
      ...sheetData.timestamps.map((ts) => {
        const value = row[ts]
        if (value === 'Present' || value === 'p') return 'p'
        return 'a'
      }),
    ])

    const ws = XLSX.utils.aoa_to_sheet([header, ...body])
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'Attendance')
    const fileSafe = group.replace(/[^A-Za-z0-9_-]+/g, '_')
    XLSX.writeFile(wb, `Attendance_${fileSafe}.xlsx`)
    setGroupStatus(`Excel downloaded for ${group}.`)
  }

  function exportDataset() {
    const attendanceByDate = {}
    const groupSheets = {}

    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i)
      if (!key) continue

      if (key.startsWith(ATT_PREFIX)) {
        attendanceByDate[key.replace(ATT_PREFIX, '')] = readJSON(key, [])
      }

      if (key.startsWith(GROUP_ATT_PREFIX)) {
        const group = groupFromStorageKey(key)
        groupSheets[group] = readJSON(key, { timestamps: [], rows: [] })
      }
    }

    const payload = {
      version: 1,
      exportedAt: new Date().toISOString(),
      students,
      attendanceByDate,
      groupSheets,
    }

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `smartface_dataset_${new Date().toISOString().replace(/[:.]/g, '-')}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    setDatasetStatus('Dataset exported successfully.')
  }

  async function handleDatasetImportChange(event) {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const text = await file.text()
      const parsed = JSON.parse(text)

      if (!parsed || typeof parsed !== 'object') {
        throw new Error('Invalid dataset format.')
      }

      const importedStudents = Array.isArray(parsed.students) ? parsed.students : []
      const importedAttendanceByDate = parsed.attendanceByDate && typeof parsed.attendanceByDate === 'object'
        ? parsed.attendanceByDate
        : {}
      const importedGroupSheets = parsed.groupSheets && typeof parsed.groupSheets === 'object'
        ? parsed.groupSheets
        : {}

      const proceed = window.confirm('Import dataset into existing data? This will merge data and keep previous records.')
      if (!proceed) return

      // Merge students by (name, group) and merge descriptor arrays.
      const existingStudents = readJSON(STUDENTS_KEY, [])
      const studentMap = new Map()
      const studentKey = (s) => `${(s.name || '').trim().toLowerCase()}|${(s.group || '').trim().toLowerCase()}`

      existingStudents.forEach((s) => {
        studentMap.set(studentKey(s), {
          name: s.name || '',
          group: s.group || '',
          descriptors: Array.isArray(s.descriptors) ? s.descriptors.slice() : [],
        })
      })

      importedStudents.forEach((s) => {
        const key = studentKey(s)
        const existing = studentMap.get(key)
        const incomingDesc = Array.isArray(s.descriptors) ? s.descriptors : []
        if (!existing) {
          studentMap.set(key, {
            name: s.name || '',
            group: s.group || '',
            descriptors: incomingDesc.slice(),
          })
        } else {
          existing.descriptors = existing.descriptors.concat(incomingDesc)
          studentMap.set(key, existing)
        }
      })

      const mergedStudents = Array.from(studentMap.values())
      writeJSON(STUDENTS_KEY, mergedStudents)

      // Merge attendance by date and de-duplicate by Name+Group+Time.
      const dateSet = new Set(Object.keys(importedAttendanceByDate))
      for (let i = 0; i < localStorage.length; i += 1) {
        const key = localStorage.key(i)
        if (key && key.startsWith(ATT_PREFIX)) dateSet.add(key.replace(ATT_PREFIX, ''))
      }

      const mergedAttendanceByDate = {}
      dateSet.forEach((date) => {
        const existingRows = readJSON(ATT_PREFIX + date, [])
        const incomingRows = Array.isArray(importedAttendanceByDate[date]) ? importedAttendanceByDate[date] : []
        const rowMap = new Map()

        existingRows.forEach((r) => {
          const k = `${r.Name || ''}|${r.Group || ''}|${r.Time || ''}`
          rowMap.set(k, r)
        })
        incomingRows.forEach((r) => {
          const k = `${r.Name || ''}|${r.Group || ''}|${r.Time || ''}`
          rowMap.set(k, r)
        })

        const mergedRows = Array.from(rowMap.values())
        mergedAttendanceByDate[date] = mergedRows
        writeJSON(ATT_PREFIX + date, mergedRows)
      })

      // Merge group sheets by group name, union timestamps, merge row cells.
      const groupSet = new Set(Object.keys(importedGroupSheets))
      for (let i = 0; i < localStorage.length; i += 1) {
        const key = localStorage.key(i)
        if (key && key.startsWith(GROUP_ATT_PREFIX)) groupSet.add(groupFromStorageKey(key))
      }

      groupSet.forEach((groupName) => {
        const existingSheet = readJSON(groupStorageKey(groupName), { timestamps: [], rows: [] })
        const incomingSheet = importedGroupSheets[groupName] && typeof importedGroupSheets[groupName] === 'object'
          ? importedGroupSheets[groupName]
          : { timestamps: [], rows: [] }

        const timestamps = Array.from(
          new Set([...(existingSheet.timestamps || []), ...(incomingSheet.timestamps || [])]),
        ).sort()

        const rowMap = new Map()
        ;(existingSheet.rows || []).forEach((row) => {
          rowMap.set(row.Name, { ...row })
        })
        ;(incomingSheet.rows || []).forEach((row) => {
          const prev = rowMap.get(row.Name) || { Name: row.Name }
          rowMap.set(row.Name, { ...prev, ...row })
        })

        const mergedRows = Array.from(rowMap.values())
          .map((row) => {
            const out = { Name: row.Name }
            timestamps.forEach((ts) => {
              const val = row[ts]
              out[ts] = val === 'p' || val === 'Present' ? 'p' : 'a'
            })
            return out
          })
          .sort((a, b) => (a.Name || '').localeCompare(b.Name || ''))

        writeJSON(groupStorageKey(groupName), { timestamps, rows: mergedRows })
      })

      setStudents(mergedStudents)
      const todayDate = new Date().toISOString().slice(0, 10)
      setAttendance(Array.isArray(mergedAttendanceByDate[todayDate]) ? mergedAttendanceByDate[todayDate] : [])
      setSelectedGroupView('')
      setProcessedPreview('')
      setDatasetStatus('Dataset imported and merged successfully.')
    } catch (err) {
      setDatasetStatus(`Dataset import failed: ${err.message}`)
    } finally {
      event.target.value = ''
    }
  }

  async function descriptorFromImageElement(imageEl) {
    const faceapi = getFaceApi()
    const detection = await faceapi
      .detectSingleFace(imageEl, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor()

    if (!detection) return null
    return Array.from(detection.descriptor)
  }

  async function descriptorFromCurrentVideoFrame() {
    const faceapi = getFaceApi()
    const detection = await faceapi
      .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor()

    if (!detection) return null
    return Array.from(detection.descriptor)
  }

  function addStudentSamples(name, group, descriptors) {
    const finalName = name.trim()
    const finalGroup = group.trim()
    setStudents((prev) => {
      const existing = prev.find(
        (s) => s.name.toLowerCase() === finalName.toLowerCase() && (s.group || '').toLowerCase() === finalGroup.toLowerCase(),
      )

      if (existing) {
        return prev.map((s) => {
          if (s !== existing) return s
          return { ...s, descriptors: [...(s.descriptors || []), ...descriptors] }
        })
      }

      return [...prev, { name: finalName, group: finalGroup, descriptors: descriptors.slice() }]
    })
  }

  async function handleRegisterSubmit(event) {
    event.preventDefault()

    if (!modelsReady) {
      setRegisterStatus('Models are still loading. Please wait.')
      return
    }

    const name = studentName.trim()
    if (!name || !studentPhoto) {
      setRegisterStatus('Enter name and upload one clear photo.')
      return
    }

    try {
      const imageURL = URL.createObjectURL(studentPhoto)
      const imageEl = new Image()
      imageEl.src = imageURL
      await imageEl.decode()
      const descriptor = await descriptorFromImageElement(imageEl)
      URL.revokeObjectURL(imageURL)

      if (!descriptor) {
        setRegisterStatus('No clear face detected in uploaded image.')
        return
      }

      addStudentSamples(name, studentGroup, [descriptor])
      setRegisterStatus(`Registered ${name} successfully.`)
      setStudentPhoto(null)
    } catch (err) {
      setRegisterStatus(`Registration failed: ${err.message}`)
    }
  }

  async function captureEightSamples() {
    if (!modelsReady) {
      setRegisterStatus('Models are still loading. Please wait.')
      return
    }

    const name = studentName.trim()
    if (!name) {
      setRegisterStatus('Enter student name before capture.')
      return
    }

    setIsCapturingRegister(true)
    try {
      await ensureCamera()
      const samples = []
      let attempts = 0

      while (samples.length < 8 && attempts < 32) {
        attempts += 1
        // eslint-disable-next-line no-await-in-loop
        const descriptor = await descriptorFromCurrentVideoFrame()
        if (descriptor) {
          samples.push(descriptor)
          setRegisterStatus(`Captured ${samples.length}/8 face samples...`)
        }
        // eslint-disable-next-line no-await-in-loop
        await sleep(320)
      }

      if (samples.length < 7) {
        setRegisterStatus(`Only ${samples.length} captures detected. Try again with better light.`)
        return
      }

      addStudentSamples(name, studentGroup, samples)
      setRegisterStatus(`Registered ${name} with ${samples.length} camera samples.`)
    } catch (err) {
      setRegisterStatus(`Capture failed: ${err.message}`)
    } finally {
      setIsCapturingRegister(false)
    }
  }

  async function processLiveFrame() {
    const faceapi = getFaceApi()
    const detections = await faceapi
      .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors()

    syncOverlaySize()

    const canvas = overlayRef.current
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const recognizedNow = []
    detections.forEach((det) => {
      const match = matchDescriptor(Array.from(det.descriptor), filterGroup)
      const known = match.name !== 'Unknown'
      const box = det.detection.box

      if (known) {
        markAttendance(match.name, match.group)
        recognizedNow.push(match.name)
      }

      ctx.strokeStyle = known ? '#0f9f6f' : '#dc2626'
      ctx.lineWidth = 3
      ctx.strokeRect(box.x, box.y, box.width, box.height)

      ctx.fillStyle = known ? '#0f9f6f' : '#dc2626'
      ctx.fillRect(box.x, box.y + box.height - 22, box.width, 22)

      ctx.fillStyle = '#ffffff'
      ctx.font = '13px Poppins, sans-serif'
      ctx.fillText(known ? match.name : 'Unknown', box.x + 5, box.y + box.height - 7)
    })

    setCameraStatus(recognizedNow.length ? `Live recognized: ${recognizedNow.join(', ')}` : 'Live recognized: none')
  }

  async function startLiveScanner() {
    if (!modelsReady) {
      setCameraStatus('Models are still loading. Please wait.')
      return
    }

    try {
      await ensureCamera()
      if (scannerTimerRef.current) clearInterval(scannerTimerRef.current)
      scannerTimerRef.current = setInterval(() => {
        processLiveFrame().catch((err) => setCameraStatus(`Scan error: ${err.message}`))
      }, SCAN_INTERVAL_MS)
      setCameraStatus('Live scanner started.')
    } catch (err) {
      setCameraStatus(`Camera access failed: ${err.message}`)
    }
  }

  async function handleGroupPhotoSubmit(event) {
    event.preventDefault()

    if (!modelsReady) {
      setGroupStatus('Models are still loading. Please wait.')
      return
    }

    if (!groupPhoto) {
      setGroupStatus('Upload a classroom photo first.')
      return
    }

    const group = safeGroupName(groupScopeName)
    const faceapi = getFaceApi()

    try {
      const imageURL = URL.createObjectURL(groupPhoto)
      const imageEl = new Image()
      imageEl.src = imageURL
      await imageEl.decode()

      const detections = await faceapi
        .detectAllFaces(imageEl, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors()

      const canvas = document.createElement('canvas')
      canvas.width = imageEl.width
      canvas.height = imageEl.height
      const ctx = canvas.getContext('2d')
      ctx.drawImage(imageEl, 0, 0)

      const recognized = []
      let unknown = 0

      detections.forEach((det) => {
        const match = matchDescriptor(Array.from(det.descriptor), group)
        const known = match.name !== 'Unknown'
        const box = det.detection.box

        if (known) {
          markAttendance(match.name, match.group)
          recognized.push(match.name)
        } else {
          unknown += 1
        }

        ctx.strokeStyle = known ? '#0f9f6f' : '#dc2626'
        ctx.lineWidth = 3
        ctx.strokeRect(box.x, box.y, box.width, box.height)
        ctx.fillStyle = known ? '#0f9f6f' : '#dc2626'
        ctx.fillRect(box.x, box.y + box.height - 22, box.width, 22)
        ctx.fillStyle = '#ffffff'
        ctx.font = '14px Poppins, sans-serif'
        ctx.fillText(known ? match.name : 'Unknown', box.x + 5, box.y + box.height - 6)
      })

      const presentUnique = [...new Set(recognized)]
      updateGroupMatrix(group, presentUnique, new Date().toISOString())
      setSelectedGroupView(group)

      setProcessedPreview(canvas.toDataURL('image/jpeg', 0.9))
      setGroupStatus(
        `Processed group ${group}. Recognized: ${presentUnique.length ? presentUnique.join(', ') : 'none'} | Unknown: ${unknown}`,
      )

      URL.revokeObjectURL(imageURL)
    } catch (err) {
      setGroupStatus(`Group photo processing failed: ${err.message}`)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <h1>Smart Attendence</h1>
        </div>
        <div className="topbar-actions">
          <div className={`model-pill ${modelsReady ? 'ok' : 'warn'}`}>{modelStatus}</div>
          <button type="button" className="topbar-btn" onClick={exportDataset}>
            Export Dataset
          </button>
          <button type="button" className="topbar-btn" onClick={() => datasetInputRef.current?.click()}>
            Import Dataset
          </button>
          <input
            ref={datasetInputRef}
            type="file"
            accept="application/json"
            className="hidden-file"
            onChange={handleDatasetImportChange}
          />
        </div>
      </header>

      <main className="layout">
        <section className="column-left">
          <article className="card card-register">
            <h2>Register Student</h2>
            <p className="card-note">Create robust recognition profiles with image upload or 8-shot capture.</p>
            {registerStatus && <p className="status-line">{registerStatus}</p>}

            <form onSubmit={handleRegisterSubmit} className="stack">
              <label>
                Student Name
                <input value={studentName} onChange={(e) => setStudentName(e.target.value)} placeholder="e.g. (Divya)" required />
              </label>

              <label>
                Group
                <input value={studentGroup} onChange={(e) => setStudentGroup(e.target.value)} placeholder="e.g. B2" />
              </label>

              <label>
                Upload Photo
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setStudentPhoto(e.target.files?.[0] || null)}
                  required
                />
              </label>

              <div className="button-row">
                <button type="button" className="btn btn-secondary" onClick={captureEightSamples} disabled={isCapturingRegister}>
                  {isCapturingRegister ? 'Capturing...' : 'Capture 8 Photos'}
                </button>
                <button type="submit" className="btn btn-primary">
                  Register
                </button>
              </div>
            </form>
          </article>

          <article className="card card-group">
            <h2>Group Attendance</h2>
            <p className="card-note">Process a classroom photo and mark attendance for the selected group.</p>
            {groupStatus && <p className="status-line">{groupStatus}</p>}

            <form onSubmit={handleGroupPhotoSubmit} className="stack">
              <label>
                Group Name
                <input
                  value={groupScopeName}
                  onChange={(e) => setGroupScopeName(e.target.value)}
                  placeholder="Required for group sheet"
                />
              </label>

              <label>
                Classroom Photo
                <input type="file" accept="image/*" onChange={(e) => setGroupPhoto(e.target.files?.[0] || null)} required />
              </label>

              <div className="button-row">
                <button type="submit" className="btn btn-success">
                  Process Photo
                </button>
              </div>
            </form>

            {processedPreview && <img className="preview" src={processedPreview} alt="Processed classroom preview" />}
          </article>

          <article className="card card-dataset">
            <h2>Dataset Backup</h2>
            <p className="status-line">
              Export or import full local data: students, attendance history, and group sheets.
            </p>
            {datasetStatus && <p className="status-line">{datasetStatus}</p>}
            <div className="button-row">
              <button type="button" className="btn btn-accent" onClick={exportDataset}>
                Export Dataset
              </button>
              <button type="button" className="btn btn-primary" onClick={() => datasetInputRef.current?.click()}>
                Import Dataset
              </button>
            </div>
          </article>
        </section>

        <section className="column-right">
          <article className="card live-card card-live">
            <div className="live-head">
              <h2>Live Scanner</h2>
              <p>{cameraStatus}</p>
            </div>

            <div className="live-stage">
              {!isCameraOn && <div className="camera-placeholder">Camera offline. Start scanner to begin.</div>}
              <video ref={videoRef} autoPlay muted playsInline className={isCameraOn ? 'show' : 'hide'} onLoadedMetadata={syncOverlaySize} />
              <canvas ref={overlayRef} className={isCameraOn ? 'show overlay' : 'hide overlay'} />
            </div>

            <div className="button-row">
              <button
                className={`btn ${isCameraOn ? 'btn-danger' : 'btn-primary'}`}
                onClick={isCameraOn ? stopCamera : startLiveScanner}
              >
                {isCameraOn ? 'Stop Scanner' : 'Start Scanner'}
              </button>
            </div>
          </article>

          <article className="card card-table">
            <div className="section-head">
              <h2>Today's Attendance</h2>
              <div className="inline-controls">
                <input
                  value={filterGroup}
                  onChange={(e) => setFilterGroup(e.target.value)}
                  placeholder="Filter by group"
                />
              </div>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Group</th>
                    <th>Time</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {!filteredAttendance.length && (
                    <tr>
                      <td colSpan="4" className="empty-row">
                        No records yet.
                      </td>
                    </tr>
                  )}
                  {filteredAttendance.map((row, idx) => (
                    <tr key={`${row.Name}-${row.Time}-${idx}`}>
                      <td>{row.Name}</td>
                      <td>{row.Group || '-'}</td>
                      <td>{row.Time}</td>
                      <td>
                        <span className="chip">Present</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="card card-matrix">
            <div className="section-head">
              <h2>Group-wise Attendance Sheet</h2>
              <div className="inline-controls">
                <select value={selectedGroupView} onChange={(e) => setSelectedGroupView(e.target.value)}>
                  <option value="">Select Group</option>
                  {knownGroups.map((group) => (
                    <option key={group} value={group}>
                      {group}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="btn btn-accent"
                  onClick={() => exportGroupExcel(selectedGroupView)}
                  disabled={!selectedGroupView}
                >
                  Download Excel
                </button>
              </div>
            </div>

            {!selectedGroupView && <p className="status-line">Choose a group to view present/absent matrix.</p>}

            {selectedGroupView && (
              <div className="table-wrap matrix-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>names</th>
                      {(selectedGroupSheet?.timestamps || []).map((ts) => (
                        <th key={ts}>{formatTimestampLabel(ts)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {!(selectedGroupSheet?.rows || []).length && (
                      <tr>
                        <td colSpan={(selectedGroupSheet?.timestamps?.length || 0) + 1} className="empty-row">
                          No matrix data for this group yet.
                        </td>
                      </tr>
                    )}
                    {(selectedGroupSheet?.rows || []).map((row) => (
                      <tr key={row.Name}>
                        <td>{row.Name}</td>
                        {(selectedGroupSheet?.timestamps || []).map((ts) => (
                          <td key={`${row.Name}-${ts}`}>
                            <span className={row[ts] === 'p' ? 'chip ok' : 'chip bad'}>{row[ts] === 'p' ? 'p' : 'a'}</span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        </section>
      </main>
    </div>
  )
}

export default App