# SmartFace Attendance (React)

A frontend-only face-attendance app built with React + Vite.

## Features

- Register students with `name + group` using upload or `Capture 8 Photos`
- Live scanner with face boxes and labels
- Stop scanner on `Stop Scanner` button
- Group photo processing for attendance
- Today's attendance table with group filter
- Group-wise attendance matrix (`Present` / `Absent`) by timestamp
- Excel export per group (`.xlsx`)
- Offline-first storage using `localStorage`

## Tech Stack

- React 19 + Vite 8
- `face-api.js` loaded from CDN at runtime
- `xlsx` for Excel export

## Install

```bash
npm install
```

## Run (Development)

```bash
npm run dev
```

## Build (Production)

```bash
npm run build
```

## Preview Build

```bash
npm run preview
```

## Notes

- AI model loading depends on internet access to public CDNs.
- Attendance and student data are saved in browser localStorage.
- To clear all data, clear browser site storage.