/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#pragma once

#include <cstdint>

namespace gobot {

// From SDL_scancode.h
enum class KeyCode : std::uint32_t {
    Unknown     = 0,

    A           = 4,
    B           = 5,
    C           = 6,
    D           = 7,
    E           = 8,
    F           = 9,
    G           = 10,
    H           = 11,
    I           = 12,
    J           = 13,
    K           = 14,
    L           = 15,
    M           = 16,
    N           = 17,
    O           = 18,
    P           = 19,
    Q           = 20,
    R           = 21,
    S           = 22,
    T           = 23,
    U           = 24,
    V           = 25,
    W           = 26,
    X           = 27,
    Y           = 28,
    Z           = 29,

    D1          = 30, /* 1 */
    D2          = 31, /* 2 */
    D3          = 32, /* 3 */
    D4          = 33, /* 4 */
    D5          = 34, /* 5 */
    D6          = 35, /* 6 */
    D7          = 36, /* 7 */
    D8          = 37, /* 8 */
    D9          = 38, /* 9 */
    D0          = 39, /* 0 */

    Enter       = 40,
    Escape      = 41,
    Backspace   = 42,
    Tab         = 43,
    Space       = 44,

    Minus       = 45,  /* - */
    Equal       = 46,  /* = */
    LeftBracket = 47,  /* [ */
    RightBracket = 48, /* ] */
    Backslash   = 49,  /* \ */
    Nonushash   = 50,
    Semicolon   = 51,  /* ; */
    Apostrophe  = 52,  /* ' */
    GraveAccent = 53,  /* ` */
    Comma       = 54,  /* , */
    Period      = 55,  /* . */
    Slash       = 56,  /* / */

    CapsLock    = 57,

    F1          = 58,
    F2          = 59,
    F3          = 60,
    F4          = 61,
    F5          = 62,
    F6          = 63,
    F7          = 64,
    F8          = 65,
    F9          = 66,
    F10         = 67,
    F11         = 68,
    F12         = 69,

    PrintScreen = 70,
    ScrollLock  = 71,
    Pause       = 72,
    Insert      = 73,

    Home        = 74,
    PageUp      = 75,
    Delete      = 76,
    End         = 77,
    PageDown    = 78,
    Right       = 79,
    Left        = 80,
    Down        = 81,
    Up          = 82,
    NumLock     = 83, /**< num lock on PC, clear on Mac keyboards> **/

    KP_Divide    = 84,
    KP_Multiply  = 85,
    KP_Minus     = 86,
    KP_Plus      = 87,
    KP_Enter     = 88,
    KP_1         = 89,
    KP_2         = 90,
    KP_3         = 91,
    KP_4         = 92,
    KP_5         = 93,
    KP_6         = 94,
    KP_7         = 95,
    KP_8         = 96,
    KP_9         = 97,
    KP_0         = 98,
    KP_Period    = 99,

    NonusBackSlash = 100,
    Application = 101,
    Power       = 102,
    KPEquals    = 103,
    F13         = 104,
    F14         = 105,
    F15         = 106,
    F16         = 107,
    F17         = 108,
    F18         = 109,
    F19         = 110,
    F20         = 111,
    F21         = 112,
    F22         = 113,
    F23         = 114,
    F24         = 115,
    Execute     = 116,
    Help        = 117,    /**< AL Integrated Help Center */
    Menu        = 118,    /**< Menu (show menu) */
    Select      = 119,
    Stop        = 120,    /**< AC Stop */
    Again       = 121,   /**< AC Redo/Repeat */
    Undo        = 122,    /**< AC Undo */
    Cut         = 123,     /**< AC Cut */
    Copy        = 124,    /**< AC Copy */
    Paste       = 125,   /**< AC Paste */
    Find        = 126,    /**< AC Find */
    Mute        = 127,
    VolumeUp    = 128,
    VolumeDown  = 129,
    /* not sure whether there's a reason to enable these */
/*     SDL_SCANCODE_LOCKINGCAPSLOCK = 130,  */
/*     SDL_SCANCODE_LOCKINGNUMLOCK = 131, */
/*     SDL_SCANCODE_LOCKINGSCROLLLOCK = 132, */
    KP_Comma        = 133,
    KP_Equalsas400  = 134,

    International1 = 135, /**< used on Asian keyboards, see
                                            footnotes in USB doc */
    International2 = 136,
    International3 = 137, /**< Yen */
    International4 = 138,
    International5 = 139,
    International6 = 140,
    International7 = 141,
    International8 = 142,
    International9 = 143,
    Lang1          = 144, /**< Hangul/English toggle */
    Lang2          = 145, /**< Hanja conversion */
    Lang3          = 146, /**< Katakana */
    Lang4          = 147, /**< Hiragana */
    Lang5          = 148, /**< Zenkaku/Hankaku */
    Lang6          = 149, /**< reserved */
    Lang7          = 150, /**< reserved */
    Lang8          = 151, /**< reserved */
    Lang9          = 152, /**< reserved */

    AltErase       = 153,    /**< Erase-Eaze */
    SysReq         = 154,
    Cancel         = 155,      /**< AC Cancel */
    Clear          = 156,
    Prior          = 157,
    Enter2         = 158,
    Separator      = 159,
    Out            = 160,
    Oper           = 161,
    ClearAgain     = 162,
    Crsel          = 163,
    Exsel          = 164,

    KP_00               = 176,
    KP_000              = 177,
    ThousandsSeparator  = 178,
    DecimalSeparator    = 179,
    CurrencyUnit        = 180,
    CurrencySubunit     = 181,
    KP_LeftParen        = 182,
    KP_RightParen       = 183,
    KP_LeftBrace        = 184,
    KP_RightBrace       = 185,
    KP_Tab              = 186,
    KP_BackSpace        = 187,
    KP_A                = 188,
    KP_B                = 189,
    KP_C                = 190,
    KP_D                = 191,
    KP_E                = 192,
    KP_F                = 193,
    KP_Xor              = 194,
    KP_Power            = 195,
    KP_Percent          = 196,
    KP_Less             = 197,
    KP_Greater          = 198,
    KP_Ampersand        = 199,
    KP_DBLAmpersand     = 200,
    KP_VerticalBar      = 201,
    KP_DBLVerticalBar   = 202,
    KP_Colon            = 203,
    KP_Hash             = 204,
    KP_Space            = 205,
    KP_At               = 206,
    KP_Exclam           = 207,
    KP_MemStore         = 208,
    KP_MemRecall        = 209,
    KP_MemClear         = 210,
    KP_MemAdd           = 211,
    KP_MemSubtract      = 212,
    KP_MemMultiply      = 213,
    KP_MemDivide        = 214,
    KP_PlusMinus        = 215,
    KP_Clear            = 216,
    KP_ClearEntry       = 217,
    KP_Binary           = 218,
    KP_Octal            = 219,
    KP_Decimal          = 220,
    KP_Hexadeciaml      = 221,

    LeftCtrl            = 224,
    LeftShift           = 225,
    LeftAlt             = 226, /**< alt, option */
    LeftSuper           = 227, /**< windows, command (apple), meta */
    RightCtrl           = 228,
    RightShift          = 229,
    RightAlt            = 230, /**< alt gr, option */
    RightSuper          = 231, /**< windows, command (apple), meta */

    Mode = 257,        /**< I'm not sure if this is really not covered
                                 *   by any of the above, but since there's a
                                 *   special KMOD_MODE for it I'm adding it here
                                 */

    AudioNext           = 258,
    AudioPrev           = 259,
    AudioStop           = 260,
    AudioPlay           = 261,
    AudioMute           = 262,
    MediaSelect         = 263,
    WWW                 = 264,             /**< AL Internet Browser */
    Mail                = 265,
    Calculator          = 266,      /**< AL Calculator */
    Computer            = 267,
    AC_Search           = 268,       /**< AC Search */
    AC_Home             = 269,         /**< AC Home */
    AC_Back             = 270,         /**< AC Back */
    AC_Forward          = 271,      /**< AC Forward */
    AC_Stop             = 272,         /**< AC Stop */
    AC_Refresh          = 273,      /**< AC Refresh */
    AC_BookMarks        = 274,    /**< AC Bookmarks */

    BrightNessDown      = 275,
    BrightNessUp        = 276,
    DisplaySwitch       = 277, /**< display mirroring/dual display
                                           switch, video mode switch */
    KBDILLUMToggle      = 278,
    KBDILLUMDown        = 279,
    KBDILLUMUp          = 280,
    Eject               = 281,
    Sleep               = 282,           /**< SC System Sleep */

    APP1                = 283,
    APP2                = 284,

    AudioRewind         = 285,
    AudioFastForward    = 286,

    SoftLeft            = 287, /**< Usually situated below the display on phones and
                                      used as a multi-function feature key for selecting
                                      a software defined function shown on the bottom left
                                      of the display. */
    SoftRight           = 288, /**< Usually situated below the display on phones and
                                       used as a multi-function feature key for selecting
                                       a software defined function shown on the bottom right
                                       of the display. */
    Call                = 289, /**< Used for accepting phone calls. */
    EndCall             = 290, /**< Used for rejecting phone calls. */

    /* @} *//* Mobile keys */

    /* Add any other keys here. */

    KeyCodeMaxNum = 512 /**< not a key, just marks the number of scancodes
                                 for array bounds */
};

enum class KeyModifiers : std::uint16_t {
    None = 0x0000,
    LeftShift = 0x0001,
    RightShift = 0x0002,
    LeftCtrl = 0x0040,
    RightCtrl = 0x0080,
    LeftAlt = 0x0100,
    RightAlt = 0x0200,
    LeftSuper = 0x0400,
    RightSuper = 0x0800,
    NumLock = 0x1000,
    CapsLock = 0x2000,
    Mode = 0x4000,
    ScrollLock = 0x8000,

    Ctrl = LeftShift | RightShift,
    Shift = LeftShift | RightShift,
    Alt = LeftAlt | RightAlt,
    Super = LeftSuper | RightSuper
};

} // end of namespace gobot