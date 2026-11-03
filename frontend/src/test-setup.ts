import '@testing-library/jest-dom/vitest'

// jsdom has no object URLs (browsers do); the app creates them for PDF previews.
if (!URL.createObjectURL) URL.createObjectURL = () => 'blob:test'
if (!URL.revokeObjectURL) URL.revokeObjectURL = () => undefined
